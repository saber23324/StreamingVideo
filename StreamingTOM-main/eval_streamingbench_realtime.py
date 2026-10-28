#!/usr/bin/env python3
"""Evaluate the unmodified, training-free StreamingTOM on local StreamingBench.

The model implementation is left untouched.  This file only adapts the local
normalized StreamingBench manifest to StreamingTOM's incremental inference API.
Frames are encoded once in temporal order, and every question is answered as
soon as its causal timestamp is reached.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from decord import VideoReader, cpu

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model


OPTION_PROMPT = (
    "Select the best answer to the following multiple-choice question based on "
    "the video. Respond with only the letter (A, B, C, or D) of the correct option."
)


def source_revision() -> str | None:
    """Return a revision only when this extracted source tree owns a Git repo."""
    source_root = Path(__file__).resolve().parent
    if not (source_root / ".git").exists():
        return None
    import subprocess

    return subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()


def make_prompt(row: dict[str, Any], tokenizer) -> torch.Tensor:
    text = OPTION_PROMPT + "\n" + str(row["question"]) + "\n" + "\n".join(row["options"])
    conv = copy.deepcopy(conv_templates["qwen_1_5"])
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + text)
    conv.append_message(conv.roles[1], None)
    return tokenizer_image_token(
        conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    )


def parse_option(text: str) -> str:
    match = re.search(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", text.upper())
    return match.group(1) if match else ""


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("task_type") or "all")].append(record)

    def one(rows: list[dict[str, Any]]) -> dict[str, Any]:
        correct = sum(int(row["correct"]) for row in rows)
        return {
            "examples": len(rows),
            "correct": correct,
            "accuracy": correct / len(rows) if rows else 0.0,
        }

    return {
        "overall": one(records),
        "by_task_type": {name: one(rows) for name, rows in sorted(grouped.items())},
    }


def stable_video_seed(seed: int, video_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{video_id}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def sample_causal_stream(
    video_path: str, rows: list[dict[str, Any]], fps: str
) -> tuple[np.ndarray, np.ndarray]:
    """Build one nested, fixed-rate causal stream through the last query time."""
    reader = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(reader)
    source_fps = float(reader.get_avg_fps())
    full_duration = total_frames / source_fps
    last_query = min(full_duration, max(float(row["query_time_seconds"]) for row in rows))
    if fps == "auto":
        actual_fps = 0.2 if last_query > 1800 else 0.5
    else:
        actual_fps = float(fps)
    target_frames = max(1, min(total_frames, math.ceil(last_query * actual_fps)))
    last_index = min(
        total_frames - 1,
        max(0, math.floor(last_query * source_fps) - 1),
    )
    indices = np.linspace(0, last_index, target_frames, dtype=np.int64)
    timestamps = indices.astype(np.float64) / source_fps
    return reader.get_batch(indices).asnumpy(), timestamps


def load_model(pretrained: str):
    tokenizer, model, image_processor, _ = load_pretrained_model(
        pretrained,
        None,
        get_model_name_from_path(pretrained),
        device_map="auto",
        multimodal=True,
    )
    from streamingtom.main import streamingtom

    model.tokenizer = tokenizer
    model.fps = "auto"
    model = streamingtom(model, "llava")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return tokenizer, model, image_processor


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(Path(args.dataset_manifest).read_text(encoding="utf-8"))
    split = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    allowed = {str(video_id) for video_id in split["test"]}
    rows_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest["rows"]:
        if str(row["videoID"]) in allowed:
            rows_by_video[str(row["videoID"])].append(row)

    all_video_ids = sorted(rows_by_video)
    video_ids = all_video_ids[args.shard_index :: args.num_shards]
    if args.limit_videos:
        video_ids = video_ids[: args.limit_videos]

    tokenizer, model, image_processor = load_model(args.pretrained)
    torch.cuda.reset_peak_memory_stats()
    records: list[dict[str, Any]] = []
    sampled_frames = 0

    for video_number, video_id in enumerate(video_ids, start=1):
        video_seed = stable_video_seed(args.seed, video_id)
        torch.manual_seed(video_seed)
        torch.cuda.manual_seed_all(video_seed)
        rows = sorted(
            rows_by_video[video_id],
            key=lambda row: (float(row["query_time_seconds"]), str(row["question_id"])),
        )
        frames, timestamps = sample_causal_stream(rows[0]["video_path"], rows, args.fps)
        sampled_frames += len(frames)
        prior_prefix = 0
        first_call = True

        rows_by_prefix: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            prefix = max(
                1,
                int(
                    np.searchsorted(
                        timestamps,
                        float(row["query_time_seconds"]) + 1e-6,
                        side="right",
                    )
                ),
            )
            rows_by_prefix[prefix].append(row)

        for prefix in sorted(rows_by_prefix):
            new_frames = frames[prior_prefix:prefix]
            if len(new_frames):
                pixel_values = image_processor.preprocess(
                    new_frames, return_tensors="pt"
                )["pixel_values"].half()
            else:
                pixel_values = None
            question_rows = rows_by_prefix[prefix]
            questions = [
                {"input_ids": make_prompt(row, tokenizer)} for row in question_rows
            ]
            with torch.inference_mode():
                answers = model.generate_with_streamingtom_streaming(
                    pixel_values,
                    questions,
                    video_id=video_id,
                    is_start=first_call,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                )
            first_call = False
            prior_prefix = prefix
            for row, answer in zip(question_rows, answers):
                token_ids = answer["answer"]
                text = tokenizer.decode(
                    token_ids[0].detach().cpu().tolist(), skip_special_tokens=True
                ).strip()
                prediction = parse_option(text)
                target = str(row["answer"]).strip().upper()
                records.append(
                    {
                        "videoID": video_id,
                        "question_id": row["question_id"],
                        "task_type": row.get("task_type"),
                        "query_time_seconds": row["query_time_seconds"],
                        "sampled_prefix_frames": prefix,
                        "target": target,
                        "response": text,
                        "prediction": prediction,
                        "correct": prediction == target,
                    }
                )

        pipeline = model._video_pipelines.pop(video_id, None)
        if pipeline is not None:
            pipeline.streamingtom_core.clear_cache(video_id)
        del frames
        torch.cuda.empty_cache()
        partial = summarize(records)["overall"]
        print(
            f"video={video_number}/{len(video_ids)} id={video_id} "
            f"frames={len(timestamps)} questions={len(rows)} "
            f"running={partial['correct']}/{partial['examples']}",
            flush=True,
        )

    result = summarize(records)
    result.update(
        {
            "dataset": manifest.get("dataset_name"),
            "manifest": str(Path(args.dataset_manifest).resolve()),
            "split_file": str(Path(args.split_file).resolve()),
            "pretrained": args.pretrained,
            "train_free": True,
            "wrapper": "official_streamingtom_ctr_oqm",
            "source_tree": str(Path(__file__).resolve().parent),
            "repo_commit": source_revision(),
            "fps": args.fps,
            "sampling": "single_nested_causal_stream",
            "sampled_frames": sampled_frames,
            "summed_question_prefix_frames": sum(
                int(record["sampled_prefix_frames"]) for record in records
            ),
            "test_videos": len(video_ids),
            "seed": args.seed,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "max_new_tokens": args.max_new_tokens,
            "streamingtom_config": {
                name: os.environ.get(name)
                for name in (
                    "CTR_RETAIN_TOKENS",
                    "CTR_SIMILARITY_THRESHOLD",
                    "CTR_K",
                    "CTR_BETA",
                    "OQM_RETRIEVAL_MAX_TOKENS",
                    "OQM_ENABLE_QUANTIZATION",
                    "OQM_QUANTIZATION_BITS",
                    "OQM_GROUP_SIZE",
                    "OQM_INIT_TOKEN_COUNT",
                    "OQM_SLIDING_WINDOW_SIZE",
                    "STREAMING_ENCODER_BATCH_SIZE",
                    "STREAMINGTOM_USE_FULL_PROMPT",
                )
            },
            "cuda_peak_allocated_mib": {
                str(index): torch.cuda.max_memory_allocated(index) / 1024**2
                for index in range(torch.cuda.device_count())
            },
            "cuda_peak_reserved_mib": {
                str(index): torch.cuda.max_memory_reserved(index) / 1024**2
                for index in range(torch.cuda.device_count())
            },
            "records": records,
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))
    return result


def merge(args: argparse.Namespace) -> dict[str, Any]:
    inputs = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.merge_inputs]
    records = [record for item in inputs for record in item["records"]]
    result = summarize(records)
    first = inputs[0]
    result.update(
        {
            key: first[key]
            for key in (
                "dataset",
                "manifest",
                "split_file",
                "pretrained",
                "train_free",
                "wrapper",
                "fps",
                "sampling",
                "seed",
                "max_new_tokens",
                "streamingtom_config",
            )
        }
    )
    result["source_tree"] = str(Path(__file__).resolve().parent)
    result["repo_commit"] = source_revision()
    result["sampled_frames"] = sum(item["sampled_frames"] for item in inputs)
    result["summed_question_prefix_frames"] = sum(
        int(record["sampled_prefix_frames"]) for record in records
    )
    result["test_videos"] = sum(item["test_videos"] for item in inputs)
    result["shards"] = [
        {
            "path": str(Path(path).resolve()),
            "shard_index": item["shard_index"],
            "cuda_peak_allocated_mib": item["cuda_peak_allocated_mib"],
            "cuda_peak_reserved_mib": item["cuda_peak_reserved_mib"],
        }
        for path, item in zip(args.merge_inputs, inputs)
    ]
    result["records"] = sorted(
        records, key=lambda row: (str(row["videoID"]), str(row["question_id"]))
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest")
    parser.add_argument("--split-file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--pretrained", default="lmms-lab/llava-onevision-qwen2-7b-ov")
    parser.add_argument("--fps", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument("--merge-inputs", nargs="+")
    args = parser.parse_args()
    if args.merge_inputs:
        return args
    if not args.dataset_manifest or not args.split_file:
        parser.error("--dataset-manifest and --split-file are required for evaluation")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid shard configuration")
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.merge_inputs:
        merge(cli_args)
    else:
        evaluate(cli_args)
