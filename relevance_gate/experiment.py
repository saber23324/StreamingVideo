"""Train only a tiny gate, freeze validation calibration, and evaluate held-out pairs."""

import argparse
import copy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score

from .calibration import calibrate, probabilities
from .evaluation import evaluate_records, failure_examples
from .feature_cache import FeatureDataset, collate_features
from .manifest import file_sha256, read_manifest, write_json, write_jsonl
from .models import make_model
from .protocols import TRUSTED, WEAK, target_rows, protocol_scope
from .localization import validate_local_target, localization_metrics


def load_inputs(manifest_path, index_path):
    rows = read_manifest(manifest_path)
    index = json.loads(Path(index_path).read_text())
    if index["manifest_sha256"] != file_sha256(manifest_path):
        raise ValueError("cache index belongs to a different manifest revision")
    return rows, index


def loader(rows, index, batch_size=64, shuffle=False):
    return DataLoader(FeatureDataset(rows, index), batch_size=batch_size, shuffle=shuffle,
                       collate_fn=collate_features, num_workers=0)


@torch.no_grad()
def score(model, batches, device="cpu"):
    model.eval()
    records = []
    for batch in batches:
        logits, valid, local = model(batch["visual"].to(device), batch["audio"].to(device),
                                     batch["visual_mask"].to(device), batch["audio_mask"].to(device))
        valid &= batch["valid_aux"].to(device)
        if not torch.isfinite(logits).all():
            raise ValueError("gate emitted nonfinite logits")
        for index, row in enumerate(batch["rows"]):
            record = {**row, "raw_score": float(logits[index]), "valid_aux": bool(valid[index]),
                      "quality_reason": batch["quality_reasons"][index]}
            if local is not None:
                record["conditional_local_scores"] = local[index, :batch['visual_mask'][index].sum()].cpu().tolist()
            records.append(record)
    return records


def apply_calibration(records, calibration):
    scores = probabilities([r["raw_score"] for r in records], [r["valid_aux"] for r in records], calibration)
    return [{**row, "match_score": float(value), "accepted": bool(row["valid_aux"] and value >= calibration["threshold"]),
             "score_kind": "weak_pairing_calibrated_score" if calibration.get("label_protocol") == WEAK else "validation_calibrated_probability"}
            for row, value in zip(records, scores)]


def train(args):
    rows, index = load_inputs(args.manifest, args.cache_index)
    protocol = getattr(args, "label_protocol", TRUSTED)
    rows = target_rows(rows, protocol)
    if index["encoder_metadata"]["kind"] != "native":
        raise ValueError("G1/G2/G3 must use native visual and BEATs features")
    selected = [r for r in rows if r["visible_correspondence"] is not None]
    train_rows = [r for r in selected if r["split"] == "train"]
    val_rows = [r for r in selected if r["split"] == "val"]
    for split_rows in (train_rows, val_rows):
        if {r["visible_correspondence"] for r in split_rows} != {0, 1}:
            raise ValueError("training and validation each need trusted positives AND negatives; unknowns cannot be relabeled")
    if args.output_dir.exists():
        raise FileExistsError("use a fresh training directory; legacy checkpoints are never overwritten")
    sample = FeatureDataset(train_rows, index)[0][1]
    spec = dict(visual_dim=sample["visual"].shape[-1], audio_dim=sample["audio"].shape[-1], hidden=args.hidden, dropout=args.dropout)
    if args.kind == "g3":
        if sample["visual"].shape[0] != 196:
            raise ValueError("G3 needs the verified native 196-token layout, not pooled features")
        if any(not row.get("local_token_target") for row in train_rows if row["visible_correspondence"] == 1):
            raise ValueError("G3 requires defensible token/mask targets; attention alone is not localization supervision")
        for row in train_rows + val_rows:
            if row["visible_correspondence"] == 1:
                validate_local_target(row, index["encoder_metadata"])
        spec["prototypes"] = args.prototypes
    args.output_dir.mkdir(parents=True, exist_ok=False)
    run_reports = []
    for seed in [int(value) for value in args.seeds.split(",")]:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = make_model(args.kind, spec).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
        train_batches = loader(train_rows, index, args.batch_size, True)
        val_batches = loader(val_rows, index, args.batch_size)
        best_ap, best_state, history = -1., None, []
        started = time.perf_counter()
        for epoch in range(args.epochs):
            model.train()
            total_loss, examples = 0., 0
            for batch in train_batches:
                optimizer.zero_grad(set_to_none=True)
                logits, valid, local = model(batch["visual"].to(args.device), batch["audio"].to(args.device),
                                             batch["visual_mask"].to(args.device), batch["audio_mask"].to(args.device))
                valid &= batch["valid_aux"].to(args.device)
                labels = torch.tensor([r["visible_correspondence"] for r in batch["rows"]], device=args.device, dtype=torch.float32)
                if not valid.any():
                    continue
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[valid], labels[valid])
                if local is not None:
                    local_losses = []
                    for j, row in enumerate(batch["rows"]):
                        if valid[j] and labels[j] == 1 and row.get("local_token_target"):
                            target = torch.tensor(row["local_token_target"], device=args.device, dtype=torch.float32)
                            if target.shape != local[j].shape or target.sum() <= 0:
                                raise ValueError("invalid local token supervision")
                            target = target / target.sum()
                            local_losses.append(-(target * local[j].clamp_min(1e-8).log()).sum())
                    if local_losses:
                        loss = loss + args.local_loss_weight * torch.stack(local_losses).mean()
                loss.backward()
                if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    raise ValueError("nonfinite gate gradient")
                optimizer.step()
                total_loss += float(loss) * int(valid.sum()); examples += int(valid.sum())
            if not examples:
                raise ValueError("no valid labeled training features")
            val_predictions = score(model, val_batches, args.device)
            usable = [r for r in val_predictions if r["valid_aux"]]
            if {r["visible_correspondence"] for r in usable} != {0, 1}:
                raise ValueError("validation must contain valid examples from both classes")
            ap = float(average_precision_score([r["visible_correspondence"] for r in usable], [r["raw_score"] for r in usable]))
            history.append({"epoch": epoch + 1, "train_loss": total_loss / examples, "val_auprc": ap})
            if ap > best_ap:
                best_ap, best_state = ap, copy.deepcopy(model.cpu().state_dict())
                model.to(args.device)
            print(f"{args.kind} seed={seed} epoch={epoch+1}/{args.epochs} loss={total_loss/examples:.5f} val_AP={ap:.5f}", flush=True)
        model.load_state_dict(best_state)
        checkpoint = args.output_dir / f"{args.kind}_seed{seed}.pt"
        payload = {"kind": args.kind, "spec": spec, "state_dict": best_state, "seed": seed,
                   "trainable_parameters": sum(p.numel() for p in model.parameters()),
                   "manifest_sha256": file_sha256(args.manifest), "cache_index_sha256": file_sha256(args.cache_index),
                   "encoder_metadata": index["encoder_metadata"], "history": history,
                   "seconds": time.perf_counter() - started, "early_stop_metric": "val_auprc",
                   "training_label_protocol": protocol, "semantic_validation_complete": False,
                   "training_scope": "gate-only; features detached and cached; no legacy weights loaded by trainer"}
        torch.save(payload, checkpoint)
        val_predictions = score(model, val_batches, args.device)
        calibration = calibrate(val_predictions, file_sha256(checkpoint), args.target_tpr)
        write_json(checkpoint.with_suffix(".calibration.json"), calibration)
        write_jsonl(checkpoint.with_suffix(".validation.jsonl"), apply_calibration(val_predictions, calibration))
        run_reports.append({"seed": seed, "checkpoint": str(checkpoint), "sha256": file_sha256(checkpoint),
                            "parameters": payload["trainable_parameters"], "seconds": payload["seconds"], "best_val_auprc": best_ap})
    write_json(args.output_dir / "training.json", run_reports)
    return run_reports


def load_model(args, index):
    if args.kind == "g0":
        if index["encoder_metadata"]["kind"] != "imagebind":
            raise ValueError("G0 requires aligned ImageBind audio and image features")
        identity = "g0:" + file_sha256(args.cache_index)
        return make_model("g0", {}), identity
    if args.checkpoint is None:
        raise ValueError("a learned gate checkpoint is required")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if payload["kind"] != args.kind:
        raise ValueError("requested gate kind differs from checkpoint")
    if payload["encoder_metadata"] != index["encoder_metadata"]:
        raise ValueError("gate was trained on different encoder features")
    model = make_model(payload["kind"], payload["spec"])
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, file_sha256(args.checkpoint)


def infer(args):
    rows, index = load_inputs(args.manifest, args.cache_index)
    protocol = getattr(args, "label_protocol", TRUSTED)
    rows = target_rows(rows, protocol)
    selected = [r for r in rows if r["split"] == args.split] if args.split != "all" else rows
    if not selected:
        raise ValueError("requested split is empty")
    if any(r["split"] == "debug" for r in selected) and not args.raw_only:
        raise ValueError("unresolved debug partitions support raw diagnostics only; no calibration or experiment metrics")
    model, fingerprint = load_model(args, index)
    predictions = score(model.to(args.device), loader(selected, index, args.batch_size), args.device)
    if args.command == "calibrate":
        calibration = calibrate(predictions, fingerprint, args.target_tpr)
        write_json(args.output, calibration)
        return calibration
    if args.raw_only:
        write_jsonl(args.output, predictions)
        return {"raw_only": True, "pairs": len(predictions), "semantic_metrics": "not evaluated without validation calibration"}
    if args.calibration is None:
        raise ValueError("evaluation requires frozen validation calibration")
    calibration = json.loads(args.calibration.read_text())
    if calibration.get("label_protocol", TRUSTED) != protocol:
        raise ValueError("calibration label protocol differs; weak-pairing calibration is not semantic validation")
    if calibration["model_fingerprint"] != fingerprint:
        raise ValueError("calibration belongs to a different gate / G0 feature cache")
    calibration_ids = set(calibration["validation_source_videos"])
    if args.split == "test" and any(v in calibration_ids for r in predictions for v in (r["source_video_id"], r["audio_source_video_id"]) if v):
        raise ValueError("test data overlaps calibration source videos")
    predictions = apply_calibration(predictions, calibration)
    result = evaluate_records(predictions, args.bootstrap)
    if args.kind == "g3":
        result["localization_on_official_positives"] = localization_metrics(predictions)
    result.update(model_fingerprint=fingerprint, manifest_sha256=file_sha256(args.manifest),
                  calibration_sha256=file_sha256(args.calibration), split=args.split,
                  label_protocol=protocol, semantic_validation_complete=False, semantic_scope=protocol_scope(protocol))
    write_json(args.output, result)
    write_jsonl(args.output.with_suffix(".predictions.jsonl"), predictions)
    write_jsonl(args.output.with_suffix(".failures.jsonl"), failure_examples(predictions))
    plot_results(predictions, args.output.parent / (args.output.stem + "_plots"))
    return result


def plot_results(records, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve
    known = [r for r in records if r["visible_correspondence"] is not None]
    if {r["visible_correspondence"] for r in known} != {0, 1}:
        return
    directory.mkdir(parents=True, exist_ok=True)
    weak = any(row.get("label_protocol") == WEAK for row in records)
    labels = np.array([r["visible_correspondence"] for r in known])
    values = np.array([r["match_score"] for r in known])
    fpr, tpr, _ = roc_curve(labels, values)
    precision, recall, _ = precision_recall_curve(labels, values)
    for name, x, y, xlabel, ylabel in (("roc", fpr, tpr, "False positive rate", "True positive rate"),
                                        ("pr", recall, precision, "Recall", "Precision")):
        fig, ax = plt.subplots(); ax.plot(x, y); ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1))
        if weak: ax.set_title("Weak pairing; semantic review pending")
        fig.savefig(directory / (name + ".png"), dpi=160, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots()
    for label in (0, 1):
        ax.hist(values[labels == label], bins=np.linspace(0, 1, 21), alpha=.5, label=f"label={label}")
    ax.set(xlabel="Weak-pairing calibrated score" if weak else "Validation-calibrated probability", ylabel="Count"); ax.legend()
    if weak: ax.set_title("Weak pairing; semantic review pending")
    fig.savefig(directory / "scores.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(); ax.plot([0, 1], [0, 1], "--", color="gray")
    bin_ids = np.minimum((values * 10).astype(int), 9)
    occupied = [i for i in range(10) if (bin_ids == i).any()]
    ax.plot([values[bin_ids == i].mean() for i in occupied], [labels[bin_ids == i].mean() for i in occupied], "o-")
    ax.set(xlabel="Mean weak-pairing score" if weak else "Mean predicted probability",
           ylabel="Paired fraction" if weak else "Positive fraction", xlim=(0, 1), ylim=(0, 1))
    if weak: ax.set_title("Weak pairing; semantic review pending")
    fig.savefig(directory / "reliability.png", dpi=160, bbox_inches="tight"); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "calibrate", "evaluate", "score"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-index", type=Path, required=True)
    parser.add_argument("--kind", choices=("g0", "g1", "g2", "g3"), default="g1")
    parser.add_argument("--label-protocol", choices=(TRUSTED, WEAK), default=TRUSTED,
                        help="weak_pairing explicitly defers manual semantic review; metrics are provisional pairing diagnostics")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--raw-only", action="store_true")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--prototypes", type=int, default=8)
    parser.add_argument("--local-loss-weight", type=float, default=.1)
    parser.add_argument("--seeds", default="1234,2333,3407")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--target-tpr", type=float, default=.95)
    parser.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args()
    if args.command == "train":
        if args.kind == "g0" or args.output_dir is None:
            parser.error("train requires g1/g2/g3 and --output-dir")
        result = train(args)
    else:
        if args.output is None:
            parser.error("--output is required")
        if args.command == "calibrate" and args.split != "val":
            parser.error("calibration requires --split val")
        result = infer(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
