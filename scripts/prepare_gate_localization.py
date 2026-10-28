"""Attach verified native positional targets to official source masks."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relevance_gate.localization import attach_local_target, LAYOUT
from relevance_gate.manifest import read_manifest, write_json, write_jsonl, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--native-cache-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("choose a new localization manifest")
    rows = read_manifest(args.manifest, check_files=True)
    metadata = json.loads(args.native_cache_index.read_text())["encoder_metadata"]
    selected, excluded = [], []
    for row in rows:
        if row.get("pairing_target") is None:
            continue
        try:
            selected.append(attach_local_target(row, metadata))
        except ValueError as error:
            excluded.append({"pair_id": row["pair_id"], "reason": str(error)})
    write_jsonl(args.output, selected)
    write_json(args.output.with_suffix(".mapping.json"), {"source_manifest_sha256": file_sha256(args.manifest),
        "encoder_metadata_reference_sha256": file_sha256(args.native_cache_index),
        "layout": LAYOUT, "selected_pairs": len(selected), "excluded": excluded,
        "official_positive_targets": sum(bool(r.get("local_token_target")) for r in selected),
        "global_controls_remain_unverified": True})
    print(json.dumps({"localization_pairs": len(selected), "excluded": len(excluded)}), flush=True)


if __name__ == "__main__":
    main()
