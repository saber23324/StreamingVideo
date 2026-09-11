"""Build deterministic, video-grouped manifests for the local MCQA subsets.

Only rows whose media exists locally are retained.  The normalized schema is
consumed by ``streamingtom.train_adbt_videomme`` through ``--dataset-manifest``.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
from pathlib import Path

from decord import VideoReader, cpu


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def answer_letter(candidates: list[str], answer: str) -> str:
    normalized = str(answer).strip().casefold()
    for index, candidate in enumerate(candidates):
        if str(candidate).strip().casefold() == normalized:
            return LETTERS[index]
    raise ValueError(f"answer {answer!r} is not one of {candidates!r}")


def labeled_options(candidates: list[str]) -> list[str]:
    if len(candidates) > len(LETTERS):
        raise ValueError(f"too many options: {len(candidates)}")
    return [f"{LETTERS[index]}. {str(value).strip()}" for index, value in enumerate(candidates)]


def parse_timestamp(value: str) -> float:
    parts = [float(part) for part in str(value).strip().split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"unsupported timestamp: {value!r}")


def build_mlvu(snapshot: Path) -> dict:
    annotation_path = snapshot / "MLVU/json/3_ego.json"
    video_root = snapshot / "MLVU/video/3_ego"
    source = json.loads(annotation_path.read_text(encoding="utf-8"))
    rows = []
    missing = []
    per_video_index: dict[str, int] = {}
    for record in source:
        video_id = Path(record["video"]).stem
        path = video_root / record["video"]
        if not path.is_file():
            missing.append(record["video"])
            continue
        candidates = [str(value) for value in record["candidates"]]
        per_video_index[video_id] = per_video_index.get(video_id, 0) + 1
        rows.append(
            {
                "videoID": video_id,
                "video_path": str(path.resolve()),
                "question_id": f"mlvu_ego_{video_id}_{per_video_index[video_id]}",
                "question": str(record["question"]).strip(),
                "options": labeled_options(candidates),
                "answer": answer_letter(candidates, record["answer"]),
                "query_time_seconds": None,
                "task_type": "ego",
                "source_duration_seconds": float(record["duration"]),
            }
        )
    return {
        "dataset_name": "mlvu_ego_partial",
        "source_readme": str((snapshot / "README.md").resolve()),
        "source_annotation": str(annotation_path.resolve()),
        "rows": rows,
        "source_rows": len(source),
        "missing_rows": len(missing),
        "missing_video_names": sorted(set(missing)),
    }


def streaming_video_id(question_id: str) -> str:
    match = re.search(r"_sample_(\d+)_\d+$", question_id)
    if not match:
        raise ValueError(f"cannot parse StreamingBench question id: {question_id!r}")
    return f"sample_{int(match.group(1))}"


def build_streaming(root: Path) -> dict:
    annotation_path = root / "StreamingBench/Real_Time_Visual_Understanding.csv"
    video_root = root / "Real_Time_Visual_Understanding_videos"
    with annotation_path.open(encoding="utf-8-sig", newline="") as handle:
        source = list(csv.DictReader(handle))
    rows = []
    missing = []
    invalid_timestamps = []
    duration_cache: dict[str, float] = {}
    for record in source:
        video_id = streaming_video_id(record["question_id"])
        path = video_root / video_id / "video.mp4"
        if not path.is_file():
            missing.append(video_id)
            continue
        candidates = ast.literal_eval(record["options"])
        if not isinstance(candidates, list) or len(candidates) != 4:
            raise ValueError(f"invalid options for {record['question_id']}: {candidates!r}")
        query_time = parse_timestamp(record["time_stamp"])
        if video_id not in duration_cache:
            reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
            duration_cache[video_id] = len(reader) / float(reader.get_avg_fps())
        if query_time > duration_cache[video_id] + 1.0:
            invalid_timestamps.append(
                {
                    "question_id": record["question_id"],
                    "time_stamp": record["time_stamp"],
                    "parsed_seconds": query_time,
                    "video_duration_seconds": duration_cache[video_id],
                }
            )
            continue
        # The CSV already prefixes options with A./B./C./D.; retain it exactly.
        rows.append(
            {
                "videoID": video_id,
                "video_path": str(path.resolve()),
                "question_id": record["question_id"],
                "question": record["question"].strip(),
                "options": [str(value).strip() for value in candidates],
                "answer": record["answer"].strip().upper(),
                "query_time_seconds": query_time,
                "task_type": record["task_type"],
                "frames_required": record["frames_required"],
                "temporal_clue_type": record["temporal_clue_type"],
            }
        )
    return {
        "dataset_name": "streamingbench_realtime_partial",
        "source_readme": str((root / "README.md").resolve()),
        "source_annotation": str(annotation_path.resolve()),
        "rows": rows,
        "source_rows": len(source),
        "missing_rows": len(missing),
        "missing_video_names": sorted(set(missing)),
        "invalid_timestamp_rows": invalid_timestamps,
    }


def write_manifest(payload: dict, path: Path) -> None:
    video_ids = sorted({str(row["videoID"]) for row in payload["rows"]})
    payload = dict(payload)
    payload["available_rows"] = len(payload["rows"])
    payload["available_videos"] = len(video_ids)
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(payload["rows"], sort_keys=True).encode("utf-8")
    ).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"{payload['dataset_name']}: rows={payload['available_rows']}/"
        f"{payload['source_rows']} videos={payload['available_videos']} "
        f"missing_rows={payload['missing_rows']} "
        f"invalid_rows={len(payload.get('invalid_timestamp_rows', []))} -> {path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mlvu-snapshot",
        default="/home/yxd/pkt/huggingface/hub/datasets--MLVU--MVLU/"
        "snapshots/06ddc388aa34746b3abba77972b7a7dfd977f7a3",
    )
    parser.add_argument("--streaming-root", default="/home/yxd/pkt/streaming/dataset")
    parser.add_argument("--output-dir", default="results/dataset_manifests")
    parser.add_argument("--dataset", choices=("all", "mlvu", "streaming"), default="all")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if args.dataset in {"all", "mlvu"}:
        write_manifest(build_mlvu(Path(args.mlvu_snapshot)), output_dir / "mlvu_ego.json")
    if args.dataset in {"all", "streaming"}:
        write_manifest(
            build_streaming(Path(args.streaming_root)),
            output_dir / "streamingbench_realtime.json",
        )


if __name__ == "__main__":
    main()
