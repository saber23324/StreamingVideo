"""Run M1 in order, or save an honest readiness report when labels are missing.

This does not enable the gate in the VLM or start deferred GUI/IMU experiments.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relevance_gate.manifest import audit_manifest, read_manifest, file_sha256, write_json
from relevance_gate.protocols import TRUSTED, WEAK, target_rows, protocol_scope


def readiness(rows, stage="all", label_protocol=TRUSTED):
    if stage not in ("g0", "learned", "all"):
        raise ValueError("unknown M1 stage")
    semantic_hard = [r for r in rows if r["split"] == "test" and r["visible_correspondence"] == 0
                     and r["negative_type"] in ("same_class", "shifted", "offscreen")]
    rows = target_rows(rows, label_protocol)
    splits = {}
    for split in ("train", "val", "test", "debug"):
        selected = [r for r in rows if r["split"] == split]
        counts = Counter("unknown" if r["visible_correspondence"] is None else str(r["visible_correspondence"]) for r in selected)
        splits[split] = dict(counts)
    hard = [r for r in rows if r["split"] == "test" and r["visible_correspondence"] == 0
            and r["negative_type"] in ("same_class", "shifted", "offscreen")]
    stages = {}
    for name, required in (("g0", ("val", "test")), ("learned", ("train", "val", "test"))):
        missing = [f"{split}: trusted positive and negative visible-correspondence labels" for split in required
                   if not splits[split].get("0") or not splits[split].get("1")]
        if not hard and label_protocol == TRUSTED:
            missing.append("test: audited same-class / shifted / off-screen hard negatives")
        stages[name] = {"ready": not missing, "missing": missing}
    requested = ("g0", "learned") if stage == "all" else (stage,)
    missing = list(dict.fromkeys(item for name in requested for item in stages[name]["missing"]))
    return {"ready": not missing, "requested_stage": stage, "stage_readiness": stages,
            "label_protocol": label_protocol, "semantic_scope": protocol_scope(label_protocol),
            "manual_review_deferred": label_protocol == WEAK, "trusted_test_hard_negatives": len(semantic_hard),
            "splits": splits, "missing": missing, "excluded_debug_pairs": sum(r["split"] == "debug" for r in rows),
            "unknowns_are_never_relabeled": True,
            "media_requirement": "authorized, causally aligned, trusted source-disjoint subset; full AVSBench is not required for development"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--gpu", default="6")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seeds", default="1234,2333,3407")
    parser.add_argument("--stage", choices=("g0", "learned", "all"), default="all",
                        help="G0 needs labeled val/test; learned gates additionally need labeled train")
    parser.add_argument("--imagebind-index", type=Path, help="reuse a verified cache after label-only changes")
    parser.add_argument("--native-index", type=Path)
    parser.add_argument("--defer-manual-review", action="store_true",
                        help="execute an explicitly weak-pairing development run; semantic labels and M1 acceptance remain pending")
    parser.add_argument("--preflight-only", action="store_true")
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument("--legacy-id-files", type=Path, nargs="*", default=[
        repo / "results/adbt_videomme_split.json",
        repo / "results/dataset_splits/mlvu_full_mcqa_seed1234_pathgroup_80_20.json",
        repo / "results/dataset_splits/streamingbench_realtime_seed1234_80_20.json"])
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    args.work_dir = args.work_dir.resolve()
    protected_ids = set()
    protected_sources = {}
    for path in args.legacy_id_files:
        split = json.loads(path.read_text())
        protected_ids.update(str(v) for values in split.values() if isinstance(values, list) for v in values if isinstance(v, str))
        protected_sources[str(path)] = file_sha256(path)
    rows = read_manifest(args.manifest, check_files=True, legacy_ids=protected_ids)
    protocol = WEAK if args.defer_manual_review else TRUSTED
    report = {**readiness(rows, args.stage, protocol), "manifest": str(args.manifest.resolve()),
              "manifest_sha256": file_sha256(args.manifest), "M2_enabled": False,
              "data_audit": audit_manifest(rows, check_files=True, legacy_ids=protected_ids),
              "legacy_id_sources": protected_sources,
              "legacy_dedup_scope": "known source identifiers and current media hashes; cross-dataset unknown source aliases/pretraining overlap unverified",
              "purpose": "M1 independent visible audio/visual correspondence; no downstream gains assumed"}
    if args.work_dir.exists():
        raise FileExistsError("use a fresh experiment work-dir")
    args.work_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.work_dir / "readiness.json", report)
    if not report["ready"]:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(2)
    if args.preflight_only:
        print(json.dumps(report, indent=2)); return
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONNOUSERSITE="1",
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", HF_HOME="/home/yxd/.cache/huggingface", TOKENIZERS_PARALLELISM="false")
    commands = []

    def run(arguments):
        command = [sys.executable, *map(str, arguments)]
        commands.append(shlex.join(command))
        write_json(args.work_dir / "commands.json", {"commands": commands, "cuda_visible_devices": args.gpu})
        print("RUN " + shlex.join(command), flush=True)
        with (args.work_dir / f"command_{len(commands):02d}.log").open("w") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True,
                           cwd=Path(__file__).resolve().parents[1])

    indices = {}
    encoders = ("imagebind",) if args.stage == "g0" else ("native",) if args.stage == "learned" else ("imagebind", "native")
    for encoder in encoders:
        index = args.work_dir / (encoder + "_cache.json")
        arguments = ["scripts/cache_av_gate_features.py", "--manifest", args.manifest, "--encoder", encoder,
                     "--cache-dir", f"/nvme_data/pkt/streaming/dataset/relevance_gate/cache/{encoder}", "--index", index]
        reusable = args.imagebind_index if encoder == "imagebind" else args.native_index
        if reusable is not None:
            arguments.extend(["--reuse-index", reusable.resolve()])
        run(arguments)
        indices[encoder] = index
    common = ["-m", "relevance_gate.experiment"]
    label_arguments = ["--label-protocol", protocol]
    g0_report = None
    if args.stage in ("g0", "all"):
        calibration = args.work_dir / "g0.calibration.json"
        g0_report = args.work_dir / "g0.test.json"
        run([*common, "calibrate", "--kind", "g0", "--manifest", args.manifest, "--cache-index", indices["imagebind"],
             "--split", "val", "--output", calibration, *label_arguments])
        run([*common, "evaluate", "--kind", "g0", "--manifest", args.manifest, "--cache-index", indices["imagebind"],
             "--split", "test", "--calibration", calibration, "--output", g0_report, *label_arguments])
    results = []
    for kind in (("g1", "g2") if args.stage in ("learned", "all") else ()):
        directory = args.work_dir / kind
        run([*common, "train", "--kind", kind, "--manifest", args.manifest, "--cache-index", indices["native"],
             "--output-dir", directory, "--epochs", args.epochs, "--seeds", args.seeds, *label_arguments])
        for seed in args.seeds.split(","):
            checkpoint = directory / f"{kind}_seed{seed}.pt"
            output = directory / f"{kind}_seed{seed}.test.json"
            run([*common, "evaluate", "--kind", kind, "--manifest", args.manifest, "--cache-index", indices["native"],
                 "--checkpoint", checkpoint, "--calibration", checkpoint.with_suffix(".calibration.json"),
                 "--split", "test", "--output", output, *label_arguments])
            results.append({"kind": kind, "seed": int(seed), "report": str(output),
                            "metrics": json.loads(output.read_text())["overall"]})
    write_json(args.work_dir / "results.json", {"g0_report": str(g0_report) if g0_report else None, "learned_gates": results,
               "executed_stage": args.stage, "requested_experiments_executed": True, "M1_accepted": False, "M2_enabled": False,
               "label_protocol": protocol, "manual_review_deferred": args.defer_manual_review, "semantic_scope": protocol_scope(protocol),
               "next_requirement": "review hard-negative metrics/failures before frozen gate-to-AudioRouter integration"})


if __name__ == "__main__":
    main()
