"""Auditable pair manifests: unknown labels stay unknown, including corruptions."""

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


STRATA = ("positive", "cross_video", "same_class", "shifted", "silence", "offscreen", "noise")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))


def read_manifest(path, check_files=False, legacy_ids=()):
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        data = json.loads(path.read_text())
        rows = data if isinstance(data, list) else data["rows"]
    audit_manifest(rows, check_files=check_files, legacy_ids=legacy_ids)
    return rows


def audit_manifest(rows, check_files=False, legacy_ids=()):
    legacy_ids = set(legacy_ids)
    pair_ids = set()
    splits = defaultdict(set)
    content_splits = defaultdict(set)
    path_splits = defaultdict(set)
    media_hashes = {}
    verified_declared_hashes = 0
    missing = []
    legacy_overlap = set()
    labels = Counter()
    for row in rows:
        required = ("pair_id", "source_video_id", "audio_source_video_id", "split",
                    "frame_timestamp_s", "audio_start_s", "audio_end_s",
                    "visible_correspondence", "negative_type", "provenance")
        if any(key not in row for key in required):
            raise ValueError(f"missing required pair field: {row.get('pair_id')}")
        if row["pair_id"] in pair_ids:
            raise ValueError(f"duplicate pair_id: {row['pair_id']}")
        pair_ids.add(row["pair_id"])
        if row["split"] not in ("train", "val", "test", "debug"):
            raise ValueError("split must be train / val / test / debug")
        if row["split"] == "debug" and row.get("scope") not in ("development_only", "unlabeled_development_only"):
            raise ValueError("debug pairs must be explicitly scoped to development only")
        if row.get("split_provenance") in ("missing_official_source", "ambiguous_source_identifier", "official_source_crosses_splits") and row["split"] != "debug":
            raise ValueError("unresolved public sources cannot be assigned an experiment split")
        t, start, end = (float(row[key]) for key in ("frame_timestamp_s", "audio_start_s", "audio_end_s"))
        if not all(math.isfinite(v) for v in (t, start, end)) or not 0 <= start <= end <= t or end - start > 2. + 1e-6:
            raise ValueError(f"noncausal/invalid audio window: {row['pair_id']}")
        label = row["visible_correspondence"]
        if label is not None and (type(label) is not int or label not in (0, 1)):
            raise ValueError("correspondence label must be 0, 1, or null")
        if label is not None and not row.get("label_provenance"):
            raise ValueError("known semantic labels need label_provenance")
        if label is not None and row.get("label_status") not in ("official", "audited"):
            raise ValueError("generated or unreviewed pairs cannot receive semantic ground truth")
        if row["negative_type"] not in STRATA:
            raise ValueError("unknown negative_type")
        labels["unknown" if label is None else str(label)] += 1
        # Both sides of a cross-video pair must belong to the SAME split.
        for video_id in {row["source_video_id"], row["audio_source_video_id"]}:
            if video_id:
                splits[video_id].add(row["split"])
                if video_id in legacy_ids:
                    legacy_overlap.add(video_id)
        for key in ("video_sha256", "audio_media_sha256", "source_media_sha256"):
            if row.get(key):
                content_splits[row[key]].add(row["split"])
        for key in ("frame_path", "video_path", "audio_path", "source_region_mask"):
            if check_files and row.get(key) and not Path(row[key]).is_file():
                missing.append({"pair_id": row["pair_id"], "field": key, "path": row[key]})
            if key in ("video_path", "audio_path") and row.get(key):
                path = str(Path(row[key]).resolve())
                path_splits[path].add(row["split"])
                if check_files and Path(path).is_file():
                    if path not in media_hashes:
                        media_hashes[path] = file_sha256(path)
                    content_splits[media_hashes[path]].add(row["split"])
        if check_files:
            # A source-video checksum cannot be compared with an extracted PNG.
            # Verify it only when the original video file is present in this row.
            for hash_field, path_field in (("source_media_sha256", "video_path"),
                                           ("video_sha256", "video_path"),
                                           ("audio_media_sha256", "audio_path")):
                if row.get(hash_field) and row.get(path_field) and Path(row[path_field]).is_file():
                    actual = media_hashes[str(Path(row[path_field]).resolve())]
                    if row[hash_field] != actual:
                        raise ValueError(f"declared media checksum mismatch: {row['pair_id']} {hash_field}")
                    verified_declared_hashes += 1
        if not row.get("frame_path") and not row.get("video_path"):
            raise ValueError("a frame or video path is required")
    leaked_ids = sorted(key for key, values in splits.items() if len(values) > 1)
    leaked_hashes = sorted(key for key, values in content_splits.items() if len(values) > 1)
    leaked_paths = sorted(key for key, values in path_splits.items() if len(values) > 1)
    if leaked_ids or leaked_hashes or leaked_paths:
        raise ValueError(f"source-video/content split leakage: {leaked_ids[:5]} {leaked_hashes[:5]} {leaked_paths[:5]}")
    if legacy_overlap:
        raise ValueError(f"overlap with protected legacy source videos: {sorted(legacy_overlap)[:10]}")
    if missing:
        raise ValueError(f"missing media/annotations: {missing[:5]}")
    return {"pairs": len(rows), "labels": dict(labels), "splits": dict(Counter(row["split"] for row in rows)),
            "strata": dict(Counter(row["negative_type"] for row in rows)), "source_videos": len(splits),
            "leaked_source_ids": [], "leaked_content_hashes": [], "legacy_overlap": [],
            "legacy_id_count": len(legacy_ids), "media_content_checked": check_files,
            "verified_declared_media_hashes": verified_declared_hashes}


def cache_key(row, encoder_metadata, media_checksums):
    identity = {"pair": {key: row.get(key) for key in (
        "source_video_id", "audio_source_video_id", "frame_timestamp_s", "frame_pts_s",
        "audio_start_s", "audio_end_s", "frame_path", "video_path", "audio_path", "corruption")},
        "encoders": encoder_metadata, "media": media_checksums}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
