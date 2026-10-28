"""Build/audit AVSBench manifests without downloading or inventing semantic labels."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relevance_gate.data_tools import avsbench_manifest, negative_candidates, merge_reviews
from relevance_gate.manifest import audit_manifest, read_manifest, write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("avsbench", "controls", "review", "audit"))
    parser.add_argument("--root", type=Path, default=Path("/nvme_data/pkt/streaming/dataset/avsbench"))
    parser.add_argument("--subset", choices=("s4", "ms3"), default="s4")
    parser.add_argument("--timestamps", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-id-files", type=Path, nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "avsbench":
        if args.timestamps is None:
            parser.error("--timestamps is required; filename indices are not decoder timestamps")
        times = [json.loads(line) for line in args.timestamps.read_text().splitlines() if line.strip()]
        protected = set()
        for path in args.legacy_id_files:
            split = json.loads(path.read_text())
            protected.update(str(v) for values in split.values() if isinstance(values, list) for v in values if isinstance(v, str))
        rows, audit = avsbench_manifest(args.root, args.subset, times, protected)
        write_jsonl(args.output, rows); write_json(args.output.with_suffix(".audit.json"), audit)
        print(json.dumps({key: value for key, value in audit.items() if key != "invalid"}, indent=2))
        return
    if args.manifest is None:
        parser.error("--manifest is required")
    rows = read_manifest(args.manifest, check_files=True)
    if args.command == "controls":
        rows = negative_candidates(rows, args.seed)
    elif args.command == "review":
        if args.reviews is None:
            parser.error("--reviews is required")
        rows = merge_reviews(rows, args.reviews)
    if args.command == "audit":
        write_json(args.output, audit_manifest(rows, check_files=True))
    else:
        write_jsonl(args.output, rows)
        if args.command == "controls":
            with args.output.with_suffix(".review.csv").open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=["pair_id", "visible_correspondence", "reviewer", "evidence"])
                writer.writeheader()
                writer.writerows({"pair_id": row["pair_id"], "visible_correspondence": "", "reviewer": "", "evidence": ""}
                                 for row in rows if row["visible_correspondence"] is None)


if __name__ == "__main__":
    main()
