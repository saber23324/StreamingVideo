"""Export a seeded, score-blind human-review queue after automated stages."""
import argparse
from collections import Counter, defaultdict
import csv
import html
import json
from pathlib import Path
import random
import sys
import wave

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relevance_gate.feature_cache import PairReader
from relevance_gate.manifest import file_sha256, read_manifest, write_json, write_jsonl


def select_review_rows(rows, seed=1234):
    """Selection uses no model scores, preserving a usable held-out audit set."""
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for row in rows:
        if row["split"] in ("train", "val", "test"):
            grouped[(row["split"], row["negative_type"])].append(row)
    quotas = {"positive": 10, "cross_video": 20, "same_class": 20, "shifted": 10, "silence": 5, "noise": 10, "offscreen": 10}
    selected = []
    for split in ("train", "val", "test"):
        for kind, quota in quotas.items():
            candidates = sorted(grouped[(split, kind)], key=lambda r: r["pair_id"])
            rng.shuffle(candidates)
            selected.extend(candidates[:quota])
    return selected


def select_quality_followups(rows, audio_audit, already_selected):
    flagged = set(audio_audit.get("correlation_below_095", []))
    flagged.update(k for k,v in audio_audit.get("checks", {}).items() if v.get("reason"))
    seen = {r["source_video_id"] for r in already_selected if r["visible_correspondence"] == 1}
    result = []
    for row in sorted(rows, key=lambda r: (r["source_video_id"], r["frame_timestamp_s"])):
        source = row["source_video_id"]
        if source in flagged and source not in seen and row["visible_correspondence"] == 1:
            result.append(row)
            seen.add(source)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--audio-verification", type=Path, help="add a separate source-quality follow-up list, excluded from the score-blind sample")
    args = parser.parse_args()
    rows = read_manifest(args.manifest, check_files=True)
    selected = select_review_rows(rows, args.seed)
    blind_ids = [r["pair_id"] for r in selected]
    quality_followups = []
    if args.audio_verification is not None:
        quality_followups = select_quality_followups(rows, json.loads(args.audio_verification.read_text()), selected)
        selected.extend(quality_followups)
    if args.output_dir.exists():
        raise FileExistsError("choose a new review directory; never overwrite human annotations")
    args.output_dir.mkdir(parents=True)
    reader, cards, previews = PairReader(), [], []
    for row in selected:
        image, window, timing = reader(row)
        image_name, audio_name = row["pair_id"] + ".png", row["pair_id"] + ".wav"
        image.save(args.output_dir / image_name)
        with wave.open(str(args.output_dir / audio_name), "wb") as stream:
            stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(16000)
            stream.writeframes((window.samples.clamp(-1, 1).numpy()*32767).astype(np.int16).tobytes())
        mask_html = ""
        quality_note = ""
        if window.reason in ("audio_pts_overlap", "audio_pts_gap", "missing_audio", "malformed_audio", "nonfinite_audio"):
            quality_note = '<p><strong>输入质量异常：此预览不能作为原始声音缺失的证据；语义判断无法确认时保留 unknown，先核查原始媒体。</strong></p>'
        references = '<details><summary>音频原素材（仅供质量核对）</summary>'
        references += f'<p><a href="{html.escape(row["audio_path"], quote=True)}">输入音频的来源文件</a></p>'
        if row.get("archived_audio_path"):
            references += f'<p><a href="{html.escape(row["archived_audio_path"], quote=True)}">配套归档 WAV</a></p>'
        references += '<p>语义判定仍以本页当前帧与指定历史窗口为准。</p></details>'
        if row.get("source_region_mask"):
            mask_name = row["pair_id"] + ".mask.png"
            with Image.open(row["source_region_mask"]) as mask:
                mask.save(args.output_dir / mask_name)
            mask_html = f'<details><summary>官方原始帧标注（需核查对当前因果窗是否适用）</summary><img src="{html.escape(mask_name)}"></details>'
        previews.append({"pair_id": row["pair_id"], "frame_pts_s": timing["frame_pts_s"],
                         "quality_valid": window.valid, "quality_reason": window.reason})
        cards.append(f'<article data-split="{row["split"]}"><h3>{html.escape(row["pair_id"])}</h3>'
            f'<p>分割：{row["split"]}；候选类型：{row["negative_type"]}；原始语义标签：{row["visible_correspondence"]}</p>'
            f'<img src="{html.escape(image_name)}"><audio controls preload="none" src="{html.escape(audio_name)}"></audio>'
            f'<p>帧：{row["frame_timestamp_s"]:.6f}s；历史音频：[{row["audio_start_s"]:.6f}, {row["audio_end_s"]:.6f}]s；质量：{window.reason}</p>'
            '<p>填写 1 / 0 / unknown，并记录判断依据；跨视频、同类、噪声均不预设语义负标签。</p>' + quality_note + mask_html + references + '</article>')
    write_jsonl(args.output_dir / "selected_manifest.jsonl", selected)
    write_json(args.output_dir / "selection.json", {"source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": file_sha256(args.manifest), "seed": args.seed, "selected_pairs": len(selected),
        "selection": "fixed quotas and seeded shuffle without model scores; source-level official splits retained",
        "score_blind_pair_ids": blind_ids, "quality_followup_pair_ids": [r["pair_id"] for r in quality_followups],
        "quality_followup_scope": "source consistency audit only; report separately from the score-blind sample",
        "splits": dict(Counter(r["split"] for r in selected)), "strata": dict(Counter(r["negative_type"] for r in selected)),
        "human_labels_written": 0, "previews": previews})
    with (args.output_dir / "reviews.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["pair_id", "visible_correspondence", "reviewer", "evidence"])
        writer.writeheader(); writer.writerows({"pair_id": r["pair_id"]} for r in selected)
    (args.output_dir / "review.html").write_text('<!doctype html><meta charset="utf-8"><title>最终人工审核</title>'
        '<style>body{font-family:sans-serif;margin:24px}main{display:flex;flex-wrap:wrap;gap:16px}article{width:350px;border:1px solid #ccc;padding:12px}img{max-width:100%}audio{width:100%}</style>'
        '<h1>最终人工审核清单</h1><p>自动下载、官方标注复用、弱监督开发实验和工程检查之后进行此步骤。'
        '本页不展示模型分数。请判断当前可见证据与历史音频是否对应，在 reviews.csv 中填写标签、审核者与证据。'
        '无法确定的保留 unknown。</p><main>' + ''.join(cards) + '</main>')
    print(json.dumps({"review_pairs": len(selected), "review_page": str(args.output_dir / "review.html"), "human_labels_written": 0}), flush=True)


if __name__ == "__main__":
    main()
