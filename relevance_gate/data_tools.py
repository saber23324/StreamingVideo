"""Official-layout conversion and UNLABELED negative-control candidates."""

from collections import Counter, defaultdict
import copy
import csv
import json
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image

from .manifest import audit_manifest, file_sha256


def source_id(name):
    """Group timestamp-suffixed AVS clips by their original 11-character YT ID."""
    if len(name) > 11 and name[11] == "_" and re.fullmatch(r"[A-Za-z0-9_-]{11}", name[:11]):
        return name[:11]
    return name


def public_example_metadata(source, official_rows):
    """Resolve source-level splits without guessing which AVS clip is shown.

    Public filenames sometimes omit an initial underscore. MS3 filenames can
    name several official clips, including clips in different splits. Unresolved
    sources stay in debug. Old inventory ``metadata_matches`` is not authoritative.
    """
    subset = "s4" if "/single_source/" in source["url"] else "ms3" if "/multi_sources/" in source["url"] else None
    public_id = source["id"]
    candidates = []
    for row in official_rows:
        if subset is not None and row.get("subset") != subset:
            continue
        name = row.get("name", row.get("video_id"))
        original = source_id(name)
        if original == public_id or (len(public_id) == 10 and original == "_" + public_id):
            candidates.append(dict(row))
    sources = {source_id(r.get("name", r.get("video_id"))) for r in candidates}
    splits = sorted({r["split"] for r in candidates})
    exact = [r for r in candidates if r.get("name", r.get("video_id")) in (public_id, "_" + public_id)]
    clip = exact[0].get("name", exact[0].get("video_id")) if len(exact) == 1 else public_id
    reason = ("missing_official_source" if not candidates else
              "ambiguous_source_identifier" if len(sources) != 1 else
              "official_source_crosses_splits" if len(splits) != 1 else "official_source_group")
    categories = {r.get("category", "multi_source") for r in candidates}
    return {"source_video_id": next(iter(sources)) if len(sources) == 1 else public_id,
            "clip_id": clip, "split": splits[0] if reason == "official_source_group" else "debug",
            "category": next(iter(categories)) if len(categories) == 1 else "unknown",
            "split_provenance": reason, "official_splits": splits,
            "clip_id_matched": len(exact) == 1,
            "official_metadata_candidates": candidates}


def avsbench_manifest(root, subset, timestamp_records, protected_ids=()):
    root = Path(root)
    if subset not in ("s4", "ms3"):
        raise ValueError("supported AVSBench subsets: s4 / ms3")
    folder = root / ("Single-source" if subset == "s4" else "Multi-sources")
    metadata = folder / ("s4_meta_data.csv" if subset == "s4" else "ms3_meta_data.csv")
    data_root = folder / ("s4_data" if subset == "s4" else "ms3_data")
    with metadata.open() as stream:
        official = list(csv.DictReader(stream))
    metadata_hash = file_sha256(metadata)
    grouped = defaultdict(set)
    for row in official:
        grouped[source_id(row.get("name", row.get("video_id")))].add(row["split"])
    timestamp_lookup = {(row["video_id"], int(row["frame_index"])): row for row in timestamp_records}
    if len(timestamp_lookup) != len(timestamp_records):
        raise ValueError("duplicate frame timestamp mapping")
    rows, invalid = [], []
    for metadata_row in official:
        video = metadata_row.get("name", metadata_row.get("video_id"))
        original = source_id(video)
        split = metadata_row["split"]
        category = metadata_row.get("category", "multi_source")
        if original in protected_ids or len(grouped[original]) > 1:
            invalid.append({"video_id": video, "reason": "legacy_overlap" if original in protected_ids else "official_source_split_overlap"})
            continue
        count = 1 if subset == "s4" and split == "train" else 5
        for frame_index in range(1, count + 1):
            time = timestamp_lookup.get((video, frame_index))
            if time is None:
                invalid.append({"video_id": video, "frame_index": frame_index, "reason": "missing_verified_frame_timestamp"})
                continue
            base = data_root / "visual_frames"
            frame = base / split / category / video / f"{video}_{frame_index}.png" if subset == "s4" else base / video / f"{video}.mp4_{frame_index}.png"
            mask = data_root / "gt_masks" / split / (Path(category) / video if subset == "s4" else Path(video)) / f"{video}_{frame_index}.png"
            if not mask.is_file():
                downloaded_mask = folder / "gt_masks" / split / (Path(category) / video if subset == "s4" else Path(video)) / f"{video}_{frame_index}.png"
                if downloaded_mask.is_file():
                    mask = downloaded_mask
            audio = data_root / "audio_wav" / split / (Path(category) if subset == "s4" else Path()) / f"{video}.wav"
            frame = Path(time.get("frame_path", frame))
            audio = Path(time.get("audio_path", audio))
            if not frame.is_file() or not mask.is_file() or not audio.is_file():
                invalid.append({"video_id": video, "frame_index": frame_index, "reason": "missing_media_or_mask"})
                continue
            with Image.open(mask) as image:
                values = np.asarray(image)
            with Image.open(frame) as image:
                size = image.size
            if values.ndim != 2 or (values.shape[1], values.shape[0]) != size:
                invalid.append({"video_id": video, "frame_index": frame_index, "reason": "mask_geometry_mismatch"})
                continue
            # A frame's causal timestamp must be supplied from actual media PTS,
            # never guessed as frame_index seconds from a PNG file name.
            timestamp = float(time["timestamp_s"])
            if not time.get("provenance"):
                raise ValueError("timestamp mapping needs decoder/extraction provenance")
            label = int((values > 0).any())
            rows.append({"pair_id": f"avsbench_{subset}_{video}_{frame_index}",
                         "source_video_id": original, "clip_id": video, "audio_source_video_id": original,
                         "split": split, "category": category, "frame_timestamp_s": timestamp, "frame_pts_s": timestamp,
                         "frame_path": str(frame.resolve()), "audio_path": str(audio.resolve()),
                         "audio_start_s": max(0., timestamp - 2.), "audio_end_s": timestamp,
                         "visible_correspondence": label, "negative_type": "positive" if label else "offscreen",
                         "source_region_mask": str(mask.resolve()), "label_status": "official",
                         "label_provenance": f"AVSBench-{subset} binary sounding-object mask; frame {frame_index}; mask sha256={file_sha256(mask)}",
                         "provenance": f"official metadata sha256={metadata_hash}; timestamp={time['provenance']}",
                         "source_media_sha256": time.get("source_media_sha256"),
                         "pairing_target": 1 if label else None,
                         "pairing_target_provenance": "original official media pairing" if label else None,
                         "annotation_scope": "official annotated visible sounding objects; empty mask does not certify all unannotated audio sources"})
    # Empty masks alone cannot certify the absence of EVERY sound source. Leave
    # these as candidates until that stronger negative claim is audited.
    for row in rows:
        if row["visible_correspondence"] == 0:
            row.update(visible_correspondence=None, label_status="pending", label_provenance=None)
    audit_manifest(rows, check_files=True)
    return rows, {"subset": subset, "official_videos": len(official), "usable_pairs": len(rows),
                  "invalid_counts": dict(Counter(r["reason"] for r in invalid)), "invalid": invalid}


def negative_candidates(rows, seed=42):
    rng = random.Random(seed)
    positives = [r for r in rows if r["negative_type"] == "positive" and r["visible_correspondence"] != 0]
    result = list(copy.deepcopy(rows))
    for row in positives:
        for mode in ("cross_video", "same_class", "shifted", "silence", "noise"):
            if mode == "same_class" and row.get("category") in (None, "unknown", "multi_source"):
                continue
            candidate = copy.deepcopy(row)
            candidate.update(pair_id=row["pair_id"] + "__" + mode, negative_type=mode,
                             visible_correspondence=None, label_status="pending", label_provenance=None,
                             source_region_mask=None, candidate_seed=seed,
                             pairing_target=None, pairing_target_provenance=None)
            candidate.pop("local_token_target", None)
            if mode in ("cross_video", "same_class"):
                pool = [r for r in positives if r["split"] == row["split"] and r["source_video_id"] != row["source_video_id"]
                        and (mode != "same_class" or r.get("category") == row.get("category"))]
                if not pool:
                    continue
                donor = rng.choice(pool)
                candidate.update(audio_source_video_id=donor["audio_source_video_id"], audio_path=donor["audio_path"],
                                 audio_donor_pair_id=donor["pair_id"], audio_media_sha256=donor.get("audio_media_sha256"))
                if mode == "cross_video":
                    candidate.update(pairing_target=0, pairing_target_provenance="different-source audio donor; weak pairing target ONLY, not semantic ground truth")
            elif mode == "shifted":
                end = max(0., float(row["audio_end_s"]) - 2.)
                if end <= 0:
                    continue
                candidate.update(audio_end_s=end, audio_start_s=max(0., end - 2.))
            elif mode == "silence":
                candidate["corruption"] = {"type": "silence"}
            elif mode == "noise":
                candidate["corruption"] = {"type": "noise", "snr_db": 0., "seed": seed}
            candidate["provenance"] += f"; generated {mode} control seed={seed}; semantic label UNKNOWN until audited"
            result.append(candidate)
    audit_manifest(result)
    return result


def freeze_reference_holdout(rows, reference_rows):
    """Expand training while keeping validation/test pairs, including donors, fixed."""
    selected = [copy.deepcopy(r) for r in rows if r["split"] == "train"]
    selected.extend(copy.deepcopy(r) for r in reference_rows if r["split"] in ("val", "test"))
    for split in ("val", "test"):
        if not any(r["split"] == split for r in selected):
            raise ValueError("reference holdout must contain both val and test")
    audit_manifest(selected)
    return selected


def merge_reviews(rows, review_csv):
    by_id = {row["pair_id"]: copy.deepcopy(row) for row in rows}
    reviewed = set()
    with Path(review_csv).open() as stream:
        for review in csv.DictReader(stream):
            pair_id = review["pair_id"]
            if pair_id not in by_id or pair_id in reviewed:
                raise ValueError("unknown or duplicate reviewed pair")
            if not review["visible_correspondence"] and not review.get("reviewer") and not review.get("evidence"):
                continue
            if not review.get("reviewer") or not review.get("evidence"):
                raise ValueError("reviews need reviewer and evidence; a generated file is not an audit")
            label = None if review["visible_correspondence"] in ("", "unknown", "null") else int(review["visible_correspondence"])
            by_id[pair_id].update(visible_correspondence=label, label_status="audited" if label is not None else "pending",
                                 label_provenance=f"reviewer={review['reviewer']}; evidence={review['evidence']}" if label is not None else None,
                                 review_provenance={"reviewer": review["reviewer"], "evidence": review["evidence"]})
            reviewed.add(pair_id)
    result = list(by_id.values())
    audit_manifest(result)
    return result
