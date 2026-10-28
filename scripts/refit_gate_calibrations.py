"""Refit validation-only calibrations without changing gate or legacy weights."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relevance_gate.experiment import infer
from relevance_gate.manifest import file_sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--g3-dir", type=Path)
    parser.add_argument("--local-manifest", type=Path)
    parser.add_argument("--local-cache-index", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="1234,2333,3407")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("choose a fresh calibration version directory")
    args.output_dir.mkdir(parents=True)
    jobs = [("g0", "g0", None, args.manifest, args.experiment_dir / "imagebind_cache.json")]
    for kind in ("g1", "g2", "g3"):
        if kind == "g3" and args.g3_dir is None:
            continue
        directory = args.g3_dir if kind == "g3" else args.experiment_dir / kind
        manifest = args.local_manifest if kind == "g3" else args.manifest
        index = args.local_cache_index if kind == "g3" else args.experiment_dir / "native_cache.json"
        for seed in args.seeds.split(","):
            jobs.append((kind+"_seed"+seed, kind, directory / f"{kind}_seed{seed}.pt", manifest, index))
    results = []
    for name, kind, checkpoint, manifest, index in jobs:
        calibration = args.output_dir / (name + ".calibration.json")
        namespace = argparse.Namespace(manifest=manifest, cache_index=index, kind=kind, checkpoint=checkpoint,
            split="val", command="calibrate", raw_only=False, label_protocol="weak_pairing", target_tpr=.95,
            device="cpu", batch_size=64, bootstrap=200, output=calibration, calibration=None)
        infer(namespace)
        namespace.command, namespace.split = "evaluate", "test"
        namespace.output, namespace.calibration = args.output_dir / (name + ".test.json"), calibration
        result = infer(namespace)
        results.append({"name": name, "checkpoint": str(checkpoint) if checkpoint else None,
            "checkpoint_sha256": file_sha256(checkpoint) if checkpoint else None,
            "report": str(namespace.output), "calibration": json.loads(calibration.read_text()),
            "overall": result["overall"], "valid_features_only": result["valid_features_only"],
            "localization": result.get("localization_on_official_positives")})
        print(name + ": " + json.dumps(result["overall"]), flush=True)
    write_json(args.output_dir / "results.json", {"label_protocol": "weak_pairing", "manual_review_pending": True,
        "semantic_validation_complete": False, "calibration_source": "validation split only", "results": results})


if __name__ == "__main__":
    main()
