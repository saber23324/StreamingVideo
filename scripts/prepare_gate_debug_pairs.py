"""Prepare an UNLABELED public-example review set and causal media previews.

This is a development asset. Temporal pairing and generated corruptions NEVER
create semantic ground truth. Official-example sources.json records the URLs.
"""
import argparse
import csv
import html
import json
from pathlib import Path
import sys
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relevance_gate.data_tools import negative_candidates, public_example_metadata
from relevance_gate.feature_cache import PairReader
from relevance_gate.manifest import audit_manifest, file_sha256, write_json, write_jsonl
from relevance_gate.media import frame_at_or_before


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=Path("/nvme_data/pkt/streaming/dataset/avsbench/public_examples/sources.json"))
    parser.add_argument("--avsbench-root", type=Path, default=Path("/nvme_data/pkt/streaming/dataset/avsbench"))
    parser.add_argument("--output-dir", type=Path, default=Path("/nvme_data/pkt/streaming/dataset/relevance_gate/public_debug"))
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("choose a new review directory")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    official_rows, metadata_hashes = [], {}
    for subset, relative in (("s4", "Single-source/s4_meta_data.csv"), ("ms3", "Multi-sources/ms3_meta_data.csv")):
        path = args.avsbench_root / relative
        with path.open() as stream:
            official_rows.extend({**row, "subset": subset} for row in csv.DictReader(stream))
        metadata_hashes[relative] = file_sha256(path)
    rows, source_audit = [], []
    for source in json.loads(args.sources.read_text()):
        if "path" not in source:
            continue
        metadata = public_example_metadata(source, official_rows)
        source_audit.append({"public_id": source["id"], "url": source["url"], **metadata})
        video_id = metadata["clip_id"]
        for timestamp in (2., 4.):
            _, actual_pts = frame_at_or_before(source["path"], timestamp)
            rows.append({"pair_id": f"public_{video_id}_t{timestamp:g}", "source_video_id": metadata["source_video_id"],
                         "clip_id": video_id, "audio_source_video_id": metadata["source_video_id"],
                         "split": metadata["split"], "category": metadata["category"],
                         "split_provenance": metadata["split_provenance"], "official_splits": metadata["official_splits"],
                         "clip_id_matched": metadata["clip_id_matched"],
                         "video_path": source["path"], "audio_path": source["path"],
                         "frame_timestamp_s": actual_pts, "audio_start_s": max(0., actual_pts - 2.), "audio_end_s": actual_pts,
                         "visible_correspondence": None, "negative_type": "positive", "label_status": "pending",
                         "label_provenance": None, "source_region_mask": None,
                         "provenance": f"official public example {source['url']}; temporal pairing ONLY, semantic label UNKNOWN",
                         "source_media_sha256": source["sha256"], "scope": "development_only"})
    rows = negative_candidates(rows)
    reader = PairReader()
    previews = []
    for row in rows:
        image, window, timing = reader(row)
        image_name, audio_name = row["pair_id"] + ".png", row["pair_id"] + ".wav"
        image.save(args.output_dir / image_name)
        with wave.open(str(args.output_dir / audio_name), "wb") as stream:
            stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(16000)
            stream.writeframes((window.samples.clamp(-1, 1).numpy() * 32767).astype(np.int16).tobytes())
        row["frame_pts_s"] = timing["frame_pts_s"]
        previews.append({"pair_id": row["pair_id"], "quality_valid": window.valid, "quality_reason": window.reason,
                         "image": image_name, "audio": audio_name})
    write_jsonl(args.output_dir / "manifest.jsonl", rows)
    write_json(args.output_dir / "source_metadata_audit.json", {"metadata_sha256": metadata_hashes, "sources": source_audit,
               "warning": "Source-level split membership does not identify the public clip or align a mask."})
    write_json(args.output_dir / "audit.json", {**audit_manifest(rows, check_files=True),
               "purpose": "unlabeled development and review; NOT an M1 experiment",
               "previews": previews, "label_warning": "No semantic labels have been asserted or inferred."})
    with (args.output_dir / "reviews.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["pair_id", "visible_correspondence", "reviewer", "evidence"])
        writer.writeheader()
        writer.writerows({"pair_id": row["pair_id"], "visible_correspondence": "", "reviewer": "", "evidence": ""} for row in rows)
    cards = []
    type_names = {"positive": "Temporally paired candidate (semantic label unverified)",
                  "cross_video": "Cross-video candidate", "same_class": "Same-category candidate",
                  "shifted": "Older-window quality control", "silence": "Silence quality control", "noise": "Noise candidate"}
    for row, preview in zip(rows, previews):
        cards.append(f'<article><h3>{html.escape(row["pair_id"])}</h3><p>Label: UNKNOWN · {html.escape(type_names.get(row["negative_type"], row["negative_type"]))}</p>'
                     f'<p>Partition: {row["split"]}; {html.escape(row["split_provenance"])}</p>'
                     f'<img width="320" src="{html.escape(preview["image"])}"><br><audio controls src="{html.escape(preview["audio"])}"></audio>'
                     f'<p>frame t={row["frame_timestamp_s"]}; audio [{row["audio_start_s"]}, {row["audio_end_s"]}]; quality={html.escape(preview["quality_reason"])}</p></article>')
    (args.output_dir / "review.html").write_text('<!doctype html><meta charset="utf-8"><title>Unlabeled AV review</title>'
        '<style>body{font-family:sans-serif;margin:24px}main{display:flex;flex-wrap:wrap;gap:24px}article{max-width:360px;border:1px solid #ccc;padding:12px}</style>'
        '<h1>Unlabeled development pairs</h1><p>Review visible correspondence separately from audio quality. Record decisions and evidence in reviews.csv. '
        'The manifest field negative_type=positive denotes a temporally paired candidate, not a positive semantic label. '
        'Debug partitions with unresolved source splits are excluded from experiments.</p><main>'
        + ''.join(cards) + '</main>')
    print(json.dumps(audit_manifest(rows), indent=2))


if __name__ == "__main__":
    main()
