"""Verify paired audio and use raw-rate tracks for causal gate extraction.

Archived 16-kHz WAVs may have been resampled from a whole clip. Truncating the
original video track before resampling gives the gate an explicit causal source.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
from scipy.signal import correlate, correlation_lags

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relevance_gate.data_tools import source_id
from relevance_gate.manifest import audit_manifest, file_sha256, read_manifest, write_json, write_jsonl
from relevance_gate.media import decode_audio


def audio_similarity(raw, archived, sample_rate=16000):
    try:
        first, first_origin, _ = decode_audio(raw)
    except ValueError as error:
        if str(error) != "overlapping/nonmonotonic audio PTS":
            raise
        return {"correlation": None, "reason": "audio_pts_overlap", "gate_quality_policy": "reject ambiguous PTS; do not silently repair timestamps"}
    second, second_origin, _ = decode_audio(archived)
    if first is None or second is None or len(first) < 2 or len(second) < 2:
        return {"correlation": None, "reason": "missing_or_empty_audio"}
    first, second = first.numpy().astype(np.float64), second.numpy().astype(np.float64)
    first -= first.mean(); second -= second.mean()
    norm = np.linalg.norm(first) * np.linalg.norm(second)
    if norm < 1e-12:
        return {"correlation": None, "reason": "silent_or_invalid_audio"}
    values = correlate(first, second, mode="full", method="fft")
    lags = correlation_lags(len(first), len(second), mode="full")
    allowed = np.abs(lags) <= int(sample_rate * .1)
    best = np.argmax(np.where(allowed, values, -np.inf))
    return {"correlation": float(values[best] / norm), "lag_seconds": float(lags[best] / sample_rate),
            "raw_origin_s": first_origin, "archived_origin_s": second_origin,
            "raw_samples": len(first), "archived_samples": len(second)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("choose a fresh causal-audio manifest")
    rows = read_manifest(args.manifest, check_files=True)
    preparation = json.loads(args.preparation.read_text())
    mapping = {source_id(r["name"]): r for r in preparation["selected_videos"]}
    if len(mapping) != len(preparation["selected_videos"]):
        raise ValueError("raw source mapping is ambiguous")
    audio_checks, result = {}, []
    for row in rows:
        source = mapping[row["audio_source_video_id"]]
        raw = source["raw_video_path"]
        if row["audio_source_video_id"] not in audio_checks:
            audio_checks[row["audio_source_video_id"]] = {"raw_video": raw, "archived_audio": row["audio_path"],
                **audio_similarity(raw, row["audio_path"])}
        updated = copy.deepcopy(row)
        updated.update(archived_audio_path=row["audio_path"], audio_path=raw,
                       audio_media_sha256=source["raw_video_sha256"],
                       audio_decode_contract="raw native-rate PTS truncation before 16-kHz resampling; causal_audio_decode_version=3; ambiguous PTS rejected")
        updated["provenance"] += "; gate audio read from paired raw video track to avoid whole-track WAV resampling lookahead"
        result.append(updated)
    audit = audit_manifest(result, check_files=True)
    write_jsonl(args.output, result)
    correlations = [v["correlation"] for v in audio_checks.values() if v.get("correlation") is not None]
    write_json(args.output.with_suffix(".audio_verification.json"), {"source_manifest_sha256": file_sha256(args.manifest),
        "output_manifest_sha256": file_sha256(args.output), "audit": audit, "checks": audio_checks,
        "min_correlation": min(correlations) if correlations else None,
        "correlation_below_095": [k for k,v in audio_checks.items() if v.get("correlation") is not None and v["correlation"] < .95],
        "scope": "automatic media consistency, not semantic correspondence labeling; raw track is authoritative gate input"})
    print(json.dumps({"causal_pairs": len(result), "source_audio_checks": len(audio_checks),
                      "min_correlation": min(correlations) if correlations else None}), flush=True)


if __name__ == "__main__":
    main()
