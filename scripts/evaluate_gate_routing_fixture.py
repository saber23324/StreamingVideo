"""Real frozen-VLM generation from an existing captured native-visual fixture.

This is bounded M2 engineering validation. Cached vision is used, so timings are
not end-to-end pipeline latency. Weak-pairing gates require explicit opt-in.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from relevance_gate.feature_cache import NativeExtractor, ImageBindExtractor, PairReader, bind_legacy_imports
from relevance_gate.manifest import file_sha256, write_json
from relevance_gate.models import make_model, ImageBindCosineGate
from relevance_gate.policy import RelevanceRoutingPolicy


def load_router(path, visual_dim, device):
    from streamingtom.audio_bottleneck import AudioConditionedBottleneck
    payload = torch.load(path, map_location="cpu", weights_only=False)
    args = payload["args"]
    router = AudioConditionedBottleneck(visual_dim=visual_dim, num_queries=args["num_queries"],
        hidden_size=args["bottleneck_hidden"], num_heads=args["num_heads"], stage=int(args.get("bottleneck_stage", 3)),
        latent_norm="none", value_mode=args["value_mode"], architecture=args["architecture"],
        attention_temperature=args["attention_temperature"])
    missing, extra = router.load_state_dict(payload["audio_bottleneck"], strict=False)
    if set(missing) - {"latent_norm.weight"} or extra:
        raise ValueError("legacy router checkpoint/interface mismatch")
    return router.to(device).eval().requires_grad_(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, default=REPO / "artifacts/relevance_gate/m0-gate-disabled-final-20261022")
    parser.add_argument("--gate-checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--audio-router", type=Path, default=REPO / "results/adbt-phase4.1-q64-tau005-epoch3-lr1e5/adbt_epoch_3.pt")
    parser.add_argument("--vision-router", type=Path, default=REPO / "results/exp01_visiononly64_train_epoch3_lr1e5/adbt_epoch_3.pt")
    parser.add_argument("--beats-checkpoint", default="/nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt")
    parser.add_argument("--allow-provisional", action="store_true")
    parser.add_argument("--g0-cache-index", type=Path)
    parser.add_argument("--g0-calibration", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("choose a new bounded integration output directory")
    args.output_dir.mkdir(parents=True)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", STREAMINGTOM_PREFIX_KV_CACHE="0")
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    bind_legacy_imports()
    from streamingtom.models.llava.llava_streamingtom import _generate_with_direct_adbt
    from transformers import AutoTokenizer

    fixture = torch.load(args.fixture_dir / "fixture.pt", map_location="cpu", weights_only=True)
    baseline = json.loads((args.fixture_dir / "report.json").read_text())
    gate_payload = torch.load(args.gate_checkpoint, map_location="cpu", weights_only=True)
    calibration = json.loads(args.calibration.read_text())
    if calibration["model_fingerprint"] != file_sha256(args.gate_checkpoint):
        raise ValueError("gate and calibration fingerprints differ")
    if calibration.get("label_protocol") == "weak_pairing" and not args.allow_provisional:
        raise ValueError("manual review pending: pass --allow-provisional for development replay")
    extractor = NativeExtractor(args.beats_checkpoint, local_tokens=gate_payload["kind"] == "g3")
    if gate_payload["encoder_metadata"] != extractor.metadata:
        raise ValueError("native encoder metadata differs from gate training")
    model, device = extractor.model, extractor.device
    model._audio_bottleneck_enabled = True
    model._direct_prefix_cache_key = None
    model._direct_prefix_cache = None
    tokenizer = AutoTokenizer.from_pretrained(extractor.metadata["pretrained"], local_files_only=True)
    visual = torch.cat([chunk["visual"] for chunk in fixture["chunks"]]).to(device)
    legacy_audio = fixture["audio_info"]["embeddings"].to(device)
    timestamps = fixture["timestamps"].to(device)
    audio_router = load_router(args.audio_router, visual.shape[-1], device)
    vision_router = load_router(args.vision_router, visual.shape[-1], device)
    gate = make_model(gate_payload["kind"], gate_payload["spec"]).to(device)
    gate.load_state_dict(gate_payload["state_dict"], strict=True)
    gate.eval().requires_grad_(False)
    reader, windows, quality, gate_images = PairReader(), [], [], []
    video_path = baseline["row"]["video_path"]
    for timestamp in timestamps.cpu().tolist():
        row = {"video_path": video_path, "audio_path": video_path, "frame_timestamp_s": timestamp,
               "audio_start_s": max(0., timestamp-2.), "audio_end_s": timestamp}
        image, window, _ = reader(row)
        gate_images.append(image)
        windows.append(window.samples)
        quality.append({"valid": window.valid, "reason": window.reason})
    with torch.inference_mode():
        gate_audio = extractor.audio.encode_windows(torch.stack(windows), device).to(device=device, dtype=torch.float32)
        original_z, original_p = audio_router(visual, legacy_audio, timestamps)
        reference_z = torch.cat([c["latents"] for c in fixture["chunks"]]).to(device)
        reference_p = torch.cat([c["attention"] for c in fixture["chunks"]]).to(device)
        if not torch.equal(original_z, reference_z) or not torch.equal(original_p, reference_p):
            raise AssertionError("original routing differs from captured real fixture")
        disabled = RelevanceRoutingPolicy(audio_router)
        dz, dp = disabled(visual, legacy_audio, timestamps)
        if not torch.equal(dz, original_z) or not torch.equal(dp, original_p):
            raise AssertionError("disabled policy differs from original")
        vision_z, vision_p = vision_router(visual, None, timestamps)
        policy = RelevanceRoutingPolicy(audio_router, vision_router, gate, calibration, enabled=True,
                                        allow_provisional_calibration=args.allow_provisional)
        valid = torch.tensor([q["valid"] for q in quality], device=device, dtype=torch.bool)
        gated_z, gated_p = policy(visual, legacy_audio, timestamps, valid_aux=valid, gate_audio=gate_audio)
        output = policy.last_gate_output
        for i, accepted in enumerate(output.accepted.tolist()):
            expected_z, expected_p = (original_z[i], original_p[i]) if accepted else (vision_z[i], vision_p[i])
            if not torch.equal(gated_z[i], expected_z) or not torch.equal(gated_p[i], expected_p):
                raise AssertionError("gate did not choose one complete original routing branch")
        missing_z, missing_p = policy(visual, None, timestamps, valid_aux=torch.zeros_like(valid))
        if not torch.equal(missing_z, vision_z) or not torch.equal(missing_p, vision_p):
            raise AssertionError("audio-missing fallback differs from explicit vision-only checkpoint")
        full = visual.reshape(-1, visual.shape[-1])
        merge_type = getattr(model.config, "mm_patch_merge_type", "flat")
        if merge_type.startswith("spatial") and getattr(model.config, "mm_newline_position", "one_token") == "one_token" and "unpad" in merge_type:
            full = torch.cat((full, model.get_model().image_newline[None].to(full)))
        variants = {"original_audio": original_z, "gate_disabled": dz, "vision_only": vision_z,
                    "provisional_gate": gated_z, "audio_missing_fallback": missing_z, "full_vision_reference": full}
        g0_diagnostics = None
        if args.g0_cache_index is not None or args.g0_calibration is not None:
            if args.g0_cache_index is None or args.g0_calibration is None:
                raise ValueError("G0 replay needs both the calibrated reference index and calibration")
            g0_index = json.loads(args.g0_cache_index.read_text())
            g0_calibration = json.loads(args.g0_calibration.read_text())
            if g0_calibration["model_fingerprint"] != "g0:" + file_sha256(args.g0_cache_index):
                raise ValueError("G0 calibration fingerprint differs from its reference encoder cache")
            imagebind = ImageBindExtractor("/nvme_data/pkt/streaming/dataset/_models/imagebind/imagebind_huge.pth", device=device)
            if imagebind.metadata != g0_index["encoder_metadata"]:
                raise ValueError("G0 encoder/preprocessing differs from calibration")
            features = []
            for i, image in enumerate(gate_images):
                _, window, _ = reader({"video_path": video_path, "audio_path": video_path,
                    "frame_timestamp_s": float(timestamps[i]), "audio_start_s": max(0., float(timestamps[i])-2.),
                    "audio_end_s": float(timestamps[i])})
                features.append(imagebind(image, window))
            iv = torch.cat([f["visual"] for f in features]).to(device)
            ia = torch.cat([f["audio"] for f in features]).to(device)
            g0_policy = RelevanceRoutingPolicy(audio_router, vision_router, ImageBindCosineGate(), g0_calibration,
                enabled=True, allow_provisional_calibration=args.allow_provisional)
            g0_z, g0_p = g0_policy(visual, legacy_audio, timestamps, valid_aux=valid, gate_visual=iv, gate_audio=ia)
            g0_output = g0_policy.last_gate_output
            for i, accept in enumerate(g0_output.accepted.tolist()):
                if not torch.equal(g0_z[i], original_z[i] if accept else vision_z[i]) or not torch.equal(g0_p[i], original_p[i] if accept else vision_p[i]):
                    raise AssertionError("G0 route differs from an original complete branch")
            variants["imagebind_gate"] = g0_z
            g0_diagnostics = {"accepted": g0_output.accepted.cpu().tolist(), "scores": g0_output.match_score.cpu().tolist(),
                              "score_kind": g0_output.diagnostics["score_kind"], "calibration_sha256": file_sha256(args.g0_calibration)}
        results = {}
        images = torch.empty((len(timestamps), 3, 384, 384))  # Only the existing cache-key shape is used.
        for name, latents in variants.items():
            cache_key = (video_path, tuple(float(t) for t in timestamps), tuple(images.shape))
            flat = latents.reshape(-1, visual.shape[-1]).detach().cpu()
            model._direct_latent_cache = {cache_key: flat}
            torch.cuda.synchronize(); started = time.perf_counter()
            tokens = _generate_with_direct_adbt(model, fixture["prompt_ids"], images,
                direct_video_path=video_path, direct_video_timestamps=timestamps,
                do_sample=False, max_new_tokens=4, use_cache=True,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
            torch.cuda.synchronize()
            results[name] = {"generated_ids": tokens.cpu().tolist(), "text": tokenizer.batch_decode(tokens, skip_special_tokens=True),
                             "visual_tokens": len(flat), "cached_generation_seconds": time.perf_counter()-started}
            print(name + ": " + json.dumps(results[name]), flush=True)
        if results["original_audio"]["generated_ids"] != fixture["generated_ids"].tolist() or results["gate_disabled"]["generated_ids"] != results["original_audio"]["generated_ids"]:
            raise AssertionError("actual generated tokens differ from original/disabled reference")
    write_json(args.output_dir / "report.json", {"purpose": "bounded M2 cached-feature engineering replay; not a downstream benchmark",
        "manual_review_deferred": args.allow_provisional, "semantic_validation_complete": False, "M1_accepted": False,
        "gate_checkpoint": str(args.gate_checkpoint), "gate_sha256": file_sha256(args.gate_checkpoint),
        "calibration_sha256": file_sha256(args.calibration), "calibration_label_protocol": calibration.get("label_protocol"),
        "fixture_sha256": file_sha256(args.fixture_dir / "fixture.pt"), "timestamps_s": timestamps.cpu().tolist(),
        "valid_aux": valid.cpu().tolist(), "quality": quality, "accepted": output.accepted.cpu().tolist(),
        "gate_scores": output.match_score.cpu().tolist(), "score_kind": output.diagnostics["score_kind"],
        "imagebind_gate": g0_diagnostics,
        "variants": results, "routing_and_disabled_generation_exact": True, "accepted_and_rejected_rows_exact": True,
        "all_parameters_frozen": all(not p.requires_grad for m in (model, audio_router, vision_router, gate) for p in m.parameters()),
        "audio_router_sha256": file_sha256(args.audio_router), "vision_router_sha256": file_sha256(args.vision_router),
        "fallback_provenance": "separate owner-trained stage-1 checkpoint; not a matched-training ablation",
        "timing_scope": "cached native vision replay plus routing and language generation; excludes full encoders/media and is not TTFT/pipeline latency"})


if __name__ == "__main__":
    main()
