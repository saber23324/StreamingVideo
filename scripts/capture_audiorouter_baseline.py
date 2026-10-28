"""Capture a small, real legacy-generation fixture before adding a gate.

This invokes the existing direct AudioRouter / stage-1 wrapper unchanged. It
does not implement a gate or compute correspondence / benchmark accuracy.
Use the existing environment and offline weights; select one held-out QA row.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=REPO / "results/dataset_manifests/streamingbench_realtime.json")
    parser.add_argument("--split", type=Path, default=REPO / "results/dataset_splits/streamingbench_realtime_seed1234_80_20.json")
    parser.add_argument("--question-id", default="Real-Time Visual Understanding_sample_123_1")
    parser.add_argument("--beats-checkpoint", type=Path, default=Path("/nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt"))
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--verify-disabled-policy", action="store_true",
                        help="wrap the router in the new default-off pass-through policy")
    parser.add_argument("--reference-dir", type=Path,
                        help="require exact equality to a previously captured real fixture")
    args = parser.parse_args()
    if args.max_frames <= 0 or args.max_new_tokens <= 0:
        parser.error("frame and generation limits must be positive for this small capture")
    return args


def main():
    args = parse_args()
    # Bind imports to this checkout, including its vendored VLM / evaluator.
    for path in reversed([REPO, REPO / "LLaVA-NeXT", REPO / "lmms-eval"]):
        sys.path.insert(0, str(path))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["STREAMINGTOM_PREFIX_KV_CACHE"] = "0"
    os.environ["BEATS_EMBEDDING_CACHE"] = ""

    import numpy as np
    import torch
    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model
    from streamingtom.main import reload_streamingtom_config
    from streamingtom.models.llava.llava_streamingtom import setup_llava_with_streamingtom
    from streamingtom.train_adbt_videomme import make_prompt, sample_video_for_rows

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this bounded real-model capture")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = checkpoint["args"]
    stage = int(saved.get("bottleneck_stage", 3))
    if stage not in (1, 3) or saved.get("value_mode") != "native":
        raise ValueError("capture requires an existing native-value stage-1 or stage-3 checkpoint")
    manifest = json.loads(args.manifest.read_text())
    split = json.loads(args.split.read_text())
    rows = [row for row in manifest["rows"] if row["question_id"] == args.question_id]
    if len(rows) != 1:
        raise ValueError("question-id must select exactly one manifest row")
    row = rows[0]
    if row["videoID"] not in split["test"] or row["videoID"] in split["train"]:
        raise ValueError("fixture must be held out in the supplied legacy split")
    if args.output_dir.exists():
        raise FileExistsError("use a fresh output-dir to preserve previous captures")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    values = {
        "CTR_SIMILARITY_THRESHOLD": "0.9", "CTR_RETAIN_TOKENS": "64",
        "CTR_K": "7", "CTR_BETA": "0.6", "OQM_RETRIEVAL_MAX_TOKENS": "12544",
        "OQM_ENABLE_QUANTIZATION": "0", "OQM_QUANTIZATION_BITS": "4",
        "OQM_GROUP_SIZE": "64", "OQM_INIT_TOKEN_COUNT": "14",
        "OQM_SLIDING_WINDOW_SIZE": "4800", "STREAMING_ENCODER_BATCH_SIZE": "4",
        "AUDIO_BOTTLENECK_STAGE": str(stage),
        "AUDIO_BOTTLENECK_NUM_QUERIES": str(saved["num_queries"]),
        "AUDIO_BOTTLENECK_HIDDEN_SIZE": str(saved["bottleneck_hidden"]),
        "AUDIO_BOTTLENECK_NUM_HEADS": str(saved["num_heads"]),
        "AUDIO_BOTTLENECK_ARCHITECTURE": saved["architecture"],
        "AUDIO_BOTTLENECK_TEMPERATURE": str(saved["attention_temperature"]),
        "AUDIO_BOTTLENECK_ALLOW_TEMPERATURE_OVERRIDE": "0",
        "AUDIO_BOTTLENECK_VALUE_MODE": "native", "AUDIO_BOTTLENECK_LATENT_NORM": "none",
        "AUDIO_BOTTLENECK_BACKEND": "direct",
        "AUDIO_BOTTLENECK_CHECKPOINT": str(args.checkpoint.resolve()),
        "AUDIO_WINDOW_SECONDS": "2.0", "AUDIO_SAMPLE_RATE": "16000",
        "AUDIO_ABLATION": "real", "BEATS_CHECKPOINT": str(args.beats_checkpoint.resolve()),
        "BEATS_BATCH_SIZE": "4", "STREAMINGTOM_MEASURE_MEMORY": "0",
    }
    os.environ.update(values)
    config = reload_streamingtom_config().config
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pretrained = saved["pretrained"]
    tokenizer, model, processor, _ = load_pretrained_model(
        pretrained, None, get_model_name_from_path(pretrained),
        device_map="auto", multimodal=True, local_files_only=True,
        attn_implementation="flash_attention_2",
    )
    model = setup_llava_with_streamingtom(model, config).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if args.verify_disabled_policy:
        from relevance_gate.policy import RelevanceRoutingPolicy
        model.audio_bottleneck = RelevanceRoutingPolicy(model.audio_bottleneck)
    if stage == 3 and model.audio_encoder.source != "beats":
        raise RuntimeError("real BEATs is required; diagnostic fallback cannot capture the baseline")
    frames, timestamps, _ = sample_video_for_rows(
        row["video_path"], [row], fps="auto", max_frames=args.max_frames
    )
    if np.any(timestamps > float(row["query_time_seconds"])):
        raise AssertionError("future visual frame in legacy fixture")
    images = processor.preprocess(frames, return_tensors="pt")["pixel_values"].half()
    prompt = make_prompt(row, tokenizer).unsqueeze(0)
    audio_info = model.encode_audio_from_video(row["video_path"], timestamps) if stage == 3 else None
    captured = []

    def capture_routing(module, inputs, kwargs, output):
        if not torch.equal(output[0], output[1] @ inputs[0]):
            raise AssertionError("native visual-value constraint changed")
        visual = inputs[0].detach().cpu()
        routed, attention = (tensor.detach().cpu() for tensor in output)
        captured.append({"visual": visual, "latents": routed, "attention": attention})

    hook = model.audio_bottleneck.register_forward_hook(capture_routing, with_kwargs=True)
    outputs = []
    durations = []
    with torch.inference_mode():
        for _ in range(2):
            # Recompute vision and routing; repeat generation must not merely hit
            # the wrapper's latent or prefix cache.
            model._direct_latent_cache = {}
            model._direct_prefix_cache = None
            model._direct_prefix_cache_key = None
            torch.cuda.synchronize()
            started = time.perf_counter()
            tokens = model.generate(
                prompt, images=[images], audio_info=audio_info,
                direct_video_path=row["video_path"], direct_video_timestamps=timestamps,
                do_sample=False, max_new_tokens=args.max_new_tokens, use_cache=True,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            )
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - started)
            outputs.append(tokens.detach().cpu())
    hook.remove()
    chunks = (len(timestamps) + 3) // 4
    if len(captured) != 2 * chunks:
        raise AssertionError("routing capture missed a legacy forward")
    exact_routing = all(
        torch.equal(captured[index][name], captured[index + chunks][name])
        for index in range(chunks) for name in ("visual", "latents", "attention")
    )
    if not exact_routing or not torch.equal(outputs[0], outputs[1]):
        raise AssertionError("repeated real generation / routing changed")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("a baseline parameter remains trainable")
    fixture = {"chunks": captured[:chunks], "prompt_ids": prompt.cpu(),
               "generated_ids": outputs[0], "timestamps": torch.as_tensor(timestamps),
               "audio_info": audio_info}
    if args.reference_dir is not None:
        reference = torch.load(args.reference_dir / "fixture.pt", map_location="cpu", weights_only=True)
        for name in ("prompt_ids", "generated_ids", "timestamps"):
            if not torch.equal(fixture[name], reference[name]):
                raise AssertionError(f"reference fixture changed: {name}")
        if len(reference["chunks"]) != len(fixture["chunks"]):
            raise AssertionError("reference chunk count changed")
        for current, old in zip(fixture["chunks"], reference["chunks"]):
            if any(not torch.equal(current[name], old[name]) for name in ("visual", "latents", "attention")):
                raise AssertionError("reference native features / routing changed")
    torch.save(fixture, args.output_dir / "fixture.pt")
    source_paths = [
        "streamingtom/audio_bottleneck/adbt.py", "streamingtom/audio_bottleneck/beats_encoder.py",
        "streamingtom/tasks/vision.py", "streamingtom/models/llava/llava_streamingtom.py",
        "streamingtom/train_adbt_videomme.py", "scripts/capture_audiorouter_baseline.py",
    ]
    if args.verify_disabled_policy:
        source_paths.extend(["relevance_gate/policy.py", "relevance_gate/models.py"])
    beats_source = getattr(sys.modules.get("BEATs"), "__file__", None)
    report = {
        "purpose": "M0 legacy generation fixture; no gate or correspondence claim",
        "repo": str(REPO), "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "source_sha256": {path: sha256(REPO / path) for path in source_paths},
        "command": shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
        "python": sys.version, "dependencies": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "datasets", "flash-attn", "numpy")},
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"), "gpu": torch.cuda.get_device_name(0),
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint),
        "beats_checkpoint_sha256": sha256(args.beats_checkpoint) if stage == 3 else None,
        "beats_source": beats_source, "beats_source_sha256": sha256(Path(beats_source)) if beats_source else None,
        "manifest_sha256": sha256(args.manifest), "split_sha256": sha256(args.split),
        "video_sha256": sha256(Path(row["video_path"])), "row": row,
        "seed": args.seed, "runtime_config": config, "fps": "auto", "max_frames": args.max_frames,
        "timestamps_s": timestamps.tolist(), "audio_windows_s": [[max(0., float(t) - 2.), float(t)] for t in timestamps] if stage == 3 else None,
        "visual_shapes": [list(chunk["visual"].shape) for chunk in captured[:chunks]],
        "routed_shapes": [list(chunk["latents"].shape) for chunk in captured[:chunks]],
        "attention_shapes": [list(chunk["attention"].shape) for chunk in captured[:chunks]],
        "audio_shape": list(audio_info["embeddings"].shape) if audio_info else None,
        "generated_ids": [output.tolist() for output in outputs],
        "generated_text": [tokenizer.batch_decode(output, skip_special_tokens=True) for output in outputs],
        "repeated_generation_exact": True, "repeated_routing_exact": exact_routing,
        "all_parameters_frozen": True, "adapter_parameters": sum(p.numel() for p in model.audio_bottleneck.parameters()),
        "generation_seconds": durations,
        "timing_scope": "diagnostic generation including vision/routing, excluding loading/audio encoding; not pipeline latency",
        "fixture_sha256": sha256(args.output_dir / "fixture.pt"),
        "gate_disabled_policy_verified": args.verify_disabled_policy,
        "reference_equivalence": str(args.reference_dir) if args.reference_dir is not None else None,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: report[key] for key in ("checkpoint_sha256", "generated_text", "visual_shapes", "routed_shapes", "repeated_generation_exact", "repeated_routing_exact")}, indent=2))


if __name__ == "__main__":
    main()
