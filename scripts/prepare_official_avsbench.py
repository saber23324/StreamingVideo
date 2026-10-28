"""Prepare source-disjoint official positives and defer generated-control review."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relevance_gate.data_tools import avsbench_manifest, negative_candidates, source_id, freeze_reference_holdout
from relevance_gate.manifest import audit_manifest, file_sha256, read_manifest, write_json, write_jsonl
from relevance_gate.official_media import align_official_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/nvme_data/pkt/streaming/dataset/avsbench"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--categories", default="dog_barking,cat_meowing,playing_piano,playing_acoustic_guitar,driving_buses,chainsawing_trees")
    parser.add_argument("--per-category", default="4,2,2", help="train,val,test source counts per category")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--reference-holdout", type=Path, help="retain reference val/test rows and audio donor identities exactly while expanding train")
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument("--legacy-id-files", type=Path, nargs="*", default=[repo / "results/adbt_videomme_split.json",
        repo / "results/dataset_splits/mlvu_full_mcqa_seed1234_pathgroup_80_20.json",
        repo / "results/dataset_splits/streamingbench_realtime_seed1234_80_20.json"])
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("choose a fresh official-media experiment directory")
    args.output_dir.mkdir(parents=True)
    metadata_path = args.root / "Single-source/s4_meta_data.csv"
    metadata = list(csv.DictReader(metadata_path.open()))
    protected = set()
    for path in args.legacy_id_files:
        split = json.loads(path.read_text())
        protected.update(str(v) for values in split.values() if isinstance(values, list) for v in values if isinstance(v, str))
    categories = args.categories.split(",")
    counts = [int(n) for n in args.per_category.split(",")]
    if len(counts) != 3 or min(counts) < 1:
        raise ValueError("three positive per-category split counts required")
    grouped = defaultdict(list)
    for row in metadata:
        if row["category"] in categories and source_id(row["name"]) not in protected:
            grouped[(row["split"], row["category"])].append(row)
    rng = random.Random(args.seed)
    times, selected, failures = [], [], []
    data = args.root / "Single-source/s4_data"
    for split, count in zip(("train", "val", "test"), counts):
        for category in categories:
            candidates = sorted(grouped[(split, category)], key=lambda r: r["name"])
            rng.shuffle(candidates)
            usable = 0
            for row in candidates:
                if usable >= count:
                    break
                video = row["name"]
                raw = data / "raw_videos" / split / category / (video + ".mp4")
                audio = data / "audio_wav" / split / category / (video + ".wav")
                frame_paths = {i: data / "visual_frames" / split / category / video / f"{video}_{i}.png"
                               for i in range(1, 2 if split == "train" else 6)}
                try:
                    if not raw.is_file() or not audio.is_file():
                        raise FileNotFoundError("missing official raw video/audio")
                    records = align_official_frames(raw, frame_paths)
                    video_hash = file_sha256(raw)
                    for record in records:
                        record.update(video_id=video, audio_path=str(audio.resolve()), source_media_sha256=video_hash,
                                      raw_video_path=str(raw.resolve()))
                    times.extend(records)
                    selected.append({**row, "raw_video_path": str(raw), "raw_video_sha256": video_hash,
                                     "audio_sha256": file_sha256(audio), "verified_frames": len(records)})
                    usable += 1
                except (ValueError, FileNotFoundError) as error:
                    failures.append({"video_id": video, "split": split, "category": category, "reason": str(error)})
            if usable < count:
                write_json(args.output_dir / "alignment_failures.json", failures)
                raise ValueError(f"insufficient pixel-verified media for {split}/{category}: {usable}/{count}")
    write_jsonl(args.output_dir / "frame_timestamps.jsonl", times)
    positives, conversion = avsbench_manifest(args.root, "s4", times, protected)
    positives = [r for r in positives if r["visible_correspondence"] == 1]
    for row in positives:
        row.update(experiment_scope="official-S4-six-category-development-subset; original official splits retained",
                   manual_review_status="official_annotation_reused; causal-window semantic spot-check deferred")
    rows = negative_candidates(positives, args.seed)
    reference = None
    if args.reference_holdout is not None:
        rows = freeze_reference_holdout(rows, read_manifest(args.reference_holdout, check_files=True, legacy_ids=protected))
        reference = {"manifest": str(args.reference_holdout.resolve()), "sha256": file_sha256(args.reference_holdout),
                     "validation_and_test_rows_unchanged": True}
    write_jsonl(args.output_dir / "manifest.jsonl", rows)
    audit = audit_manifest(rows, check_files=True, legacy_ids=protected)
    write_json(args.output_dir / "preparation.json", {"seed": args.seed, "metadata_sha256": file_sha256(metadata_path),
        "selected_videos": selected, "alignment_failures": failures, "audit": audit,
        "official_positive_pairs": len(positives), "control_pairs": len(rows)-len(positives),
        "pixel_mae_max": max(t["pixel_mae"] for t in times), "pixel_error_max": max(t["pixel_max_error"] for t in times),
        "construction_targets": dict(Counter(str(r.get("pairing_target")) for r in rows)),
        "manual_review": "deferred by user; generated controls retain unknown semantic labels",
        "frozen_holdout_reference": reference,
        "conversion_invalid_counts": conversion["invalid_counts"]})
    with (args.output_dir / "reviews.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["pair_id", "visible_correspondence", "reviewer", "evidence"])
        writer.writeheader()
        writer.writerows({"pair_id": r["pair_id"]} for r in rows if r["visible_correspondence"] is None)
    print(json.dumps({"official_positive_pairs": len(positives), "selected_videos": len(selected), "audit": audit,
                      "pixel_mae_max": max(t["pixel_mae"] for t in times)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
