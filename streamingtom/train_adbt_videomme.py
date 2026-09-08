"""Supervised adapter training on a reproducible VideoMME 80/20 split.

VideoMME is distributed as 2,700 question rows but no train split.  This
script creates a deterministic split by *videoID* so questions from one video
cannot cross train/test.  The frozen LLaVA backbone supplies projected visual
features; gradients are restricted to the audio query generator and ADBT.

This is a development split, not the published VideoMME test protocol.  The
resulting adapter can be evaluated with lmms-eval by setting
VIDEOMME_SPLIT_FILE/VIDEOMME_EVAL_SPLIT=test.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from decord import VideoReader, cpu

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model

from streamingtom.audio_bottleneck import AudioConditionedBottleneck, BEATsAudioEncoder
from streamingtom.tasks.vision import VisionTask


LOGGER = logging.getLogger("train_adbt_videomme")


def tensor_stats(tensor: torch.Tensor) -> Dict[str, object]:
    """Return cheap scalar diagnostics for a tensor without retaining a graph."""
    detached = tensor.detach()
    finite = torch.isfinite(detached)
    stats = {
        "shape": tuple(detached.shape),
        "dtype": str(detached.dtype),
        "nonfinite": int((~finite).sum().detach().cpu()),
        "numel": detached.numel(),
    }
    if finite.any():
        valid = detached[finite].float()
        stats.update(
            {
                "min": float(valid.min().cpu()),
                "max": float(valid.max().cpu()),
                "mean": float(valid.mean().cpu()),
                "std": float(valid.std(unbiased=False).cpu()),
                "max_abs": float(valid.abs().max().cpu()),
            }
        )
    else:
        stats.update({"min": None, "max": None, "mean": None, "std": None, "max_abs": None})
    return stats


def save_debug_state(
    args,
    adbt,
    optimizer,
    epoch: int,
    video_index: int,
    global_step: int,
    video_id: str,
    reason: str,
    tensors: Dict[str, torch.Tensor],
):
    """Persist the first non-finite batch so a run can be diagnosed safely."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"debug_nonfinite_step_{global_step}.pt"
    debug_tensors = {
        name: value.detach().float().cpu()
        for name, value in tensors.items()
        if torch.is_tensor(value)
    }
    payload = {
        "reason": reason,
        "epoch": epoch,
        "video_index": video_index,
        "global_step": global_step,
        "video_id": video_id,
        "tensor_stats": {name: tensor_stats(value) for name, value in tensors.items() if torch.is_tensor(value)},
        "tensors": debug_tensors,
        "audio_bottleneck": {name: value.detach().cpu() for name, value in adbt.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(payload, path)
    LOGGER.error(
        "saved non-finite debug state: reason=%s epoch=%d video=%d (%s) step=%d path=%s",
        reason,
        epoch,
        video_index,
        video_id,
        global_step,
        path,
    )
    return path


def require_finite(
    value: torch.Tensor,
    name: str,
    args,
    adbt,
    optimizer,
    epoch: int,
    video_index: int,
    global_step: int,
    video_id: str,
    tensors: Dict[str, torch.Tensor],
):
    """Stop before backward/optimizer.step can propagate NaN parameters."""
    if torch.isfinite(value).all():
        return
    save_debug_state(
        args,
        adbt,
        optimizer,
        epoch,
        video_index,
        global_step,
        video_id,
        reason=f"nonfinite_{name}",
        tensors={**tensors, name: value},
    )
    raise RuntimeError(
        f"Non-finite {name} detected at epoch={epoch} video={video_index} "
        f"({video_id}) step={global_step}; debug state was saved."
    )


def init_wandb(args, split, train_rows):
    """Create an optional W&B run without making W&B a training dependency."""
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging was requested (--wandb), but wandb is not installed. "
            "Install it with `pip install wandb` or remove --wandb."
        ) from exc

    config = vars(args).copy()
    config.update(
        {
            "dataset": "lmms-lab/Video-MME",
            "train_rows": len(train_rows),
            "train_videos": len(split["train"]),
            "test_videos": len(split["test"]),
            "gpu_visible": os.getenv("CUDA_VISIBLE_DEVICES", "all"),
        }
    )
    init_kwargs = {
        "project": args.wandb_project,
        "entity": args.wandb_entity or None,
        "name": args.wandb_run_name or None,
        "id": args.wandb_run_id or None,
        "resume": "allow",
        "mode": args.wandb_mode,
        "config": config,
    }
    # W&B does not accept None for every optional argument on all versions.
    init_kwargs = {key: value for key, value in init_kwargs.items() if value is not None}
    run = wandb.init(**init_kwargs)
    run.define_metric("train/step")
    run.define_metric("train/*", step_metric="train/step")
    run.define_metric("eval/*", step_metric="train/step")
    run.define_metric("adbt_latent/*", step_metric="train/step")
    run.define_metric("adbt_attention/*", step_metric="train/step")
    run.define_metric("adbt_query/*", step_metric="train/step")
    run.define_metric("adbt_qk/*", step_metric="train/step")
    run.summary["train_rows"] = len(train_rows)
    run.summary["train_videos"] = len(split["train"])
    run.summary["test_videos"] = len(split["test"])
    return run


def make_split(dataset, split_file: Path, seed: int = 1234, train_fraction: float = 0.8):
    video_ids = sorted(set(str(x["videoID"]) for x in dataset))
    rng = random.Random(seed)
    rng.shuffle(video_ids)
    cut = int(len(video_ids) * train_fraction)
    split = {"train": sorted(video_ids[:cut]), "test": sorted(video_ids[cut:])}
    split_file.parent.mkdir(parents=True, exist_ok=True)
    split_file.write_text(json.dumps(split, indent=2), encoding="utf-8")
    return split


def sample_video(video_path: str, fps="auto", max_frames: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    reader = VideoReader(video_path, ctx=cpu(0))
    total = len(reader)
    source_fps = float(reader.get_avg_fps())
    duration = total / source_fps
    actual_fps = (0.2 if duration > 1800 else 0.5) if fps == "auto" else float(fps)
    target = max(1, min(total, math.ceil(duration * actual_fps)))
    if max_frames > 0:
        target = min(target, max_frames)
    indices = np.linspace(0, total - 1, target, dtype=int)
    return reader.get_batch(indices).asnumpy(), indices.astype(np.float32) / source_fps


def video_path(video_id: str) -> str:
    root = Path(os.getenv("VIDEOMME_VIDEO_ROOT", "/home/yxd/.cache/huggingface/videomme/data"))
    for suffix in (".mp4", ".MP4", ".mkv"):
        candidate = root / f"{video_id}{suffix}"
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"VideoMME video not found for {video_id} below {root}")


def make_prompt(doc: Dict, tokenizer) -> torch.Tensor:
    option_prompt = (
        "Select the best answer to the following multiple-choice question based on the video. "
        "Respond with only the letter (A, B, C, or D) of the correct option."
    )
    text = option_prompt + "\n" + str(doc["question"]) + "\n" + "\n".join(doc["options"])
    conv = copy.deepcopy(conv_templates["qwen_1_5"])
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + text)
    conv.append_message(conv.roles[1], None)
    return tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")


def build_inputs(model, tokenizer, prompt_ids: torch.Tensor, answer: str, latents: torch.Tensor):
    """Insert flattened visual latents at the image marker and label answer only."""
    if prompt_ids.dim() != 1:
        prompt_ids = prompt_ids.reshape(-1)
    image_positions = torch.where(prompt_ids == IMAGE_TOKEN_INDEX)[0]
    if image_positions.numel() != 1:
        raise ValueError(f"Expected one image token, got ids={prompt_ids.tolist()}")
    image_position = int(image_positions.item())
    target = tokenizer.encode(str(answer).strip(), add_special_tokens=False)
    if not target:
        raise ValueError(f"Empty target for answer {answer!r}")
    target_ids = torch.tensor(target, dtype=torch.long, device=prompt_ids.device)
    full_ids = torch.cat([prompt_ids, target_ids])
    labels = torch.full_like(full_ids, -100)
    labels[-target_ids.numel() :] = target_ids

    embed_device = model.get_input_embeddings().weight.device
    embed_dtype = model.get_input_embeddings().weight.dtype
    prefix = full_ids[:image_position].to(embed_device)
    suffix = full_ids[image_position + 1 :].to(embed_device)
    prefix_embeds = model.get_input_embeddings()(prefix.unsqueeze(0))
    suffix_embeds = model.get_input_embeddings()(suffix.unsqueeze(0))
    latent_embeds = latents.reshape(1, -1, latents.shape[-1]).to(embed_device, dtype=embed_dtype)
    inputs_embeds = torch.cat([prefix_embeds, latent_embeds, suffix_embeds], dim=1)
    expanded_labels = torch.cat(
        [
            labels[:image_position].to(embed_device),
            torch.full((latent_embeds.shape[1],), -100, dtype=torch.long, device=embed_device),
            labels[image_position + 1 :].to(embed_device),
        ]
    ).unsqueeze(0)
    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=embed_device)
    return inputs_embeds, expanded_labels, attention_mask


def encode_visual(model, image_processor, frames: np.ndarray, batch_size: int):
    processed = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half()
    tower = model.get_vision_tower()
    projector = getattr(model.get_model(), "mm_projector", None)
    task = VisionTask(None, None)
    chunks = []
    with torch.no_grad():
        for start in range(0, processed.shape[0], batch_size):
            frame_list = list(processed[start : start + batch_size].unbind(0))
            features, _ = task.encode_vision_batch({"frames": frame_list, "grid_thw": None}, tower, projector)
            chunks.append(features)
    return torch.cat(chunks, dim=0)


def evaluate_small_set(
    model,
    tokenizer,
    image_processor,
    adbt,
    audio_encoder,
    dataset,
    split,
    max_videos: int,
    max_frames: int,
    vision_batch_size: int,
    video_offset: int = 0,
):
    """Run a small causal held-out diagnostic without building autograd graphs.

    This intentionally reports teacher-forced answer loss and first-token
    accuracy, not the official generated VideoMME metric.  It is called during
    training to detect whether the adapter is learning, while keeping the
    evaluation memory bounded to one video at a time.
    """
    all_test_ids = sorted(set(str(video_id) for video_id in split["test"]))
    # Rotate the diagnostic window.  Evaluating the same first four videos at
    # every checkpoint can make the metric look artificially frozen even when
    # the adapter changes elsewhere in the held-out split.
    if all_test_ids:
        video_offset = int(video_offset) % len(all_test_ids)
        test_ids = all_test_ids[video_offset:] + all_test_ids[:video_offset]
    else:
        test_ids = []
    if max_videos > 0:
        test_ids = test_ids[:max_videos]
    rows_by_video = {}
    for row in dataset:
        video_id = str(row["videoID"])
        if video_id in test_ids:
            rows_by_video.setdefault(video_id, []).append(row)

    model_training = model.training
    model_inner_training = getattr(model.model, "training", False)
    adbt_training = adbt.training
    model.eval()
    adbt.eval()
    loss_sum = 0.0
    first_token_correct = 0
    exact_match_correct = 0
    examples = 0
    frames_seen = 0
    try:
        with torch.no_grad():
            for video_id in test_ids:
                rows = rows_by_video.get(video_id, [])
                if not rows:
                    continue
                path = video_path(video_id)
                frames, timestamps = sample_video(path, max_frames=max_frames)
                visual = encode_visual(model, image_processor, frames, vision_batch_size)
                audio = audio_encoder.encode_video(path, timestamps, visual.device, ablation="real")
                adbt.to(device=visual.device, dtype=torch.float32)
                latents, _ = adbt(
                    visual.float(),
                    audio_embeddings=audio["embeddings"].to(visual.device, dtype=torch.float32),
                    end_timestamps=audio["end_timestamps"].to(visual.device),
                )
                frames_seen += int(frames.shape[0])
                for row in rows:
                    prompt_ids = make_prompt(row, tokenizer)
                    embeds, labels, mask = build_inputs(model, tokenizer, prompt_ids, row["answer"], latents)
                    outputs = model.model(
                        inputs_embeds=embeds,
                        attention_mask=mask,
                        use_cache=False,
                        return_dict=True,
                    )
                    shifted_labels = labels[:, 1:].to(outputs.last_hidden_state.device)
                    shifted_hidden = outputs.last_hidden_state[:, :-1]
                    answer_mask = shifted_labels.ne(-100)
                    answer_hidden = shifted_hidden[answer_mask]
                    answer_labels = shifted_labels[answer_mask]
                    answer_logits = model.lm_head(answer_hidden)
                    loss_sum += float(F.cross_entropy(answer_logits.float(), answer_labels).cpu())
                    prediction = answer_logits.argmax(dim=-1)
                    if prediction.numel() and answer_labels.numel():
                        first_token_correct += int(prediction[0].item() == answer_labels[0].item())
                    exact_match_correct += int(torch.equal(prediction, answer_labels))
                    examples += 1
                del latents, audio, visual, frames, timestamps
    finally:
        model.train(model_training)
        model.model.train(model_inner_training)
        adbt.train(adbt_training)
    return {
        "loss": loss_sum / max(1, examples),
        "first_token_accuracy": first_token_correct / max(1, examples),
        "exact_match_accuracy": exact_match_correct / max(1, examples),
        "examples": examples,
        "videos": len(test_ids),
        "frames": frames_seen,
    }


def train(args):
    if args.latent_norm is None:
        args.latent_norm = "none" if args.value_mode == "native" else "rmsnorm"
    if args.value_mode == "native" and args.latent_norm != "none":
        raise ValueError("--value-mode native requires --latent-norm none")
    if args.architecture == "phase4" and args.value_mode != "native":
        raise ValueError("--architecture phase4 requires --value-mode native")
    if args.attention_temperature <= 0:
        raise ValueError("--attention-temperature must be > 0")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = load_dataset(
        "lmms-lab/Video-MME",
        cache_dir=args.hf_cache,
        download_mode="reuse_dataset_if_exists",
    )["test"]
    split = make_split(dataset, Path(args.split_file), args.seed, 0.8)
    train_ids = set(split["train"])
    train_rows = [row for row in dataset if str(row["videoID"]) in train_ids]
    if args.max_train_samples:
        train_rows = train_rows[: args.max_train_samples]
    LOGGER.info("VideoMME rows: %d; train rows: %d; train videos: %d; test videos: %d", len(dataset), len(train_rows), len(split["train"]), len(split["test"]))
    wandb_run = init_wandb(args, split, train_rows)

    model_name = get_model_name_from_path(args.pretrained)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.pretrained, None, model_name, device_map="auto", multimodal=True
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if args.gradient_checkpointing and hasattr(model.model, "gradient_checkpointing_enable"):
        # Frozen decoder weights still need activations for dL/d(visual
        # latents). Checkpointing recomputes layer activations instead of
        # retaining the full 7B-model graph on GPU. Dropout is zero in this
        # checkpoint, so train/eval mode does not change stochastic behavior.
        model.model.gradient_checkpointing_enable()
        model.model.train()
    visual_dim = int(model.config.hidden_size)
    # Keep trainable adapter weights/optimizer state in fp32. The frozen
    # LLaVA and cached visual features remain fp16; this avoids AdamW fp16
    # overflow while preserving the inference tensor contract.
    adbt = AudioConditionedBottleneck(
        visual_dim=visual_dim,
        num_queries=args.num_queries,
        hidden_size=args.bottleneck_hidden,
        num_heads=args.num_heads,
        stage=3,
        latent_norm=args.latent_norm,
        value_mode=args.value_mode,
        architecture=args.architecture,
        attention_temperature=args.attention_temperature,
    ).to(next(model.get_vision_tower().parameters()).device, dtype=torch.float32)
    audio_encoder = BEATsAudioEncoder(
        args.beats_checkpoint,
        batch_size=args.beats_batch_size,
        output_mode="temporal" if args.architecture == "phase4" else "mean",
    )
    optimizer = torch.optim.AdamW(adbt.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.init_adapter:
        checkpoint = torch.load(args.init_adapter, map_location="cpu", weights_only=False)
        source_state = checkpoint.get("audio_bottleneck", checkpoint)
        target_state = adbt.state_dict()
        checkpoint_args = checkpoint.get("args", {})
        source_architecture = (
            checkpoint_args.get("architecture", "legacy")
            if isinstance(checkpoint_args, dict) else "legacy"
        )
        if source_architecture != args.architecture:
            raise ValueError(
                "--init-adapter architecture mismatch: "
                f"source={source_architecture} target={args.architecture}; "
                "Phase-4 must start from a Phase-4 checkpoint or fresh initialization"
            )
        source_value_mode = (
            checkpoint_args.get("value_mode", "projected")
            if isinstance(checkpoint_args, dict) else "projected"
        )
        if source_value_mode != args.value_mode:
            if source_value_mode == "projected" and args.value_mode == "native":
                dropped = sorted(set(source_state) - set(target_state))
                source_state = {
                    key: value for key, value in source_state.items() if key in target_state
                }
                LOGGER.info(
                    "migrating projected-value adapter to native-value mode; "
                    "loaded Audio Query Generator/visual Key and dropped %d legacy tensors",
                    len(dropped),
                )
            else:
                raise ValueError(
                    "unsupported --init-adapter value-mode migration: "
                    f"source={source_value_mode} target={args.value_mode}"
                )
        query_key = "query_generator.query_tokens"
        if query_key in source_state and source_state[query_key].shape != target_state[query_key].shape:
            source_queries = source_state[query_key]
            target_queries = target_state[query_key].clone()
            if (
                source_queries.ndim != 3
                or target_queries.ndim != 3
                or source_queries.shape[0] != target_queries.shape[0]
                or source_queries.shape[2] != target_queries.shape[2]
            ):
                raise ValueError(
                    "--init-adapter can resize only the query-slot dimension; "
                    f"source={tuple(source_queries.shape)} target={tuple(target_queries.shape)}"
                )
            copied_queries = min(source_queries.shape[1], target_queries.shape[1])
            target_queries[:, :copied_queries].copy_(source_queries[:, :copied_queries])
            source_state = dict(source_state)
            source_state[query_key] = target_queries
            LOGGER.info(
                "initialized %d/%d query slots from %s; %d new slots keep their independent random initialization",
                copied_queries,
                target_queries.shape[1],
                args.init_adapter,
                max(0, target_queries.shape[1] - copied_queries),
            )
        shape_mismatches = {
            key: (tuple(value.shape), tuple(target_state[key].shape))
            for key, value in source_state.items()
            if key in target_state and value.shape != target_state[key].shape
        }
        if shape_mismatches:
            raise ValueError(
                "--init-adapter is incompatible with the requested architecture; "
                f"shape mismatches={shape_mismatches}"
            )
        missing, unexpected = adbt.load_state_dict(source_state, strict=False)
        invalid_missing = [key for key in missing if not key.startswith("latent_norm.")]
        if invalid_missing or unexpected:
            raise ValueError(
                "invalid --init-adapter checkpoint: "
                f"missing={invalid_missing} unexpected={unexpected}"
            )
        LOGGER.info("initialized adapter weights from %s", args.init_adapter)

    resume_epoch, resume_video_idx, global_step = 0, 0, 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        checkpoint_args = checkpoint.get("args", {})
        checkpoint_architecture = (
            checkpoint_args.get("architecture", "legacy")
            if isinstance(checkpoint_args, dict) else "legacy"
        )
        if checkpoint_architecture != args.architecture:
            raise ValueError(
                "--resume architecture mismatch: "
                f"checkpoint={checkpoint_architecture} requested={args.architecture}"
            )
        if args.architecture == "phase4" and float(
            checkpoint_args.get("attention_temperature", 0.2)
        ) != float(args.attention_temperature):
            raise ValueError(
                "--resume attention temperature does not match the checkpoint"
            )
        checkpoint_value_mode = (
            checkpoint_args.get("value_mode", "projected")
            if isinstance(checkpoint_args, dict) else "projected"
        )
        if checkpoint_value_mode != args.value_mode:
            raise ValueError(
                "--resume value mode does not match the requested architecture: "
                f"checkpoint={checkpoint_value_mode} requested={args.value_mode}; "
                "use --init-adapter for a supported architecture migration"
            )
        missing, unexpected = adbt.load_state_dict(checkpoint["audio_bottleneck"], strict=False)
        if missing or unexpected:
            LOGGER.warning(
                "checkpoint adapter keys differ from current model (missing=%s unexpected=%s); "
                "new stability parameters keep their initialization",
                missing,
                unexpected,
            )
        if "optimizer" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
            except ValueError as exc:
                # Adding latent normalization/scale changes the parameter
                # group shape. Keep the adapter weights but start a fresh
                # optimizer rather than silently pairing states incorrectly.
                LOGGER.warning("could not restore optimizer state; using a fresh optimizer: %s", exc)
        resume_epoch = int(checkpoint.get("epoch", 0))
        resume_video_idx = int(checkpoint.get("video_idx", 0))
        global_step = int(checkpoint.get("global_step", 0))
        LOGGER.info("resuming from epoch=%d video=%d step=%d", resume_epoch, resume_video_idx, global_step)
    next_eval_step = None
    if args.eval_every_steps > 0:
        next_eval_step = ((global_step // args.eval_every_steps) + 1) * args.eval_every_steps
    optimizer.zero_grad(set_to_none=True)
    attention_collapse_streak = 0
    rows_by_video = {}
    for row in train_rows:
        rows_by_video.setdefault(str(row["videoID"]), []).append(row)
    for epoch in range(resume_epoch, args.epochs):
        video_order = list(rows_by_video)
        random.Random(args.seed + epoch).shuffle(video_order)
        if epoch == resume_epoch and resume_video_idx:
            video_order = video_order[resume_video_idx:]
        video_offset = resume_video_idx if epoch == resume_epoch else 0
        for video_idx, current_video in enumerate(video_order):
            rows = rows_by_video[current_video]
            current_video_index = video_offset + video_idx + 1
            video_loss_sum = 0.0
            video_loss_count = 0
            path = video_path(current_video)
            frames, timestamps = sample_video(path, max_frames=args.train_max_frames)
            visual = encode_visual(model, image_processor, frames, args.vision_batch_size)
            audio = audio_encoder.encode_video(path, timestamps, visual.device, ablation="real")
            adbt.to(device=visual.device, dtype=torch.float32)
            # A video has three questions in VideoMME. Reuse its frozen
            # visual/audio features, but run each language loss separately so
            # the large frozen LLM activation graph is never accumulated three
            # times on the 80-GiB cards.
            for row in rows:
                latents, _ = adbt(
                    visual.float(),
                    audio_embeddings=audio["embeddings"].to(visual.device, dtype=torch.float32),
                    end_timestamps=audio["end_timestamps"].to(visual.device),
                )
                latent_stats = tensor_stats(latents)
                adbt_stats = dict(adbt.last_diagnostics)
                attention_collapsed = (
                    adbt_stats.get("effective_support_mean", float("inf")) < 1.5
                    and adbt_stats.get("unique_argmax_mean", float("inf")) <= 2
                )
                queries_collapsed = (
                    adbt_stats.get("query_cosine_mean", -1.0) > 0.98
                )
                if attention_collapsed or queries_collapsed:
                    attention_collapse_streak += 1
                    if attention_collapse_streak == 10 or attention_collapse_streak % 100 == 0:
                        LOGGER.warning(
                            "ADBT collapse warning: streak=%d step=%d support=%.4f "
                            "unique_argmax=%.2f query_cosine=%.6f gap=%.4f",
                            attention_collapse_streak,
                            global_step,
                            adbt_stats.get("effective_support_mean", float("nan")),
                            adbt_stats.get("unique_argmax_mean", float("nan")),
                            adbt_stats.get("query_cosine_mean", float("nan")),
                            adbt_stats.get("top1_minus_top2_mean", float("nan")),
                        )
                else:
                    attention_collapse_streak = 0
                require_finite(
                    latents,
                    "latent",
                    args,
                    adbt,
                    optimizer,
                    epoch,
                    current_video_index,
                    global_step,
                    current_video,
                    tensors={"visual": visual, "audio_embeddings": audio["embeddings"]},
                )
                prompt_ids = make_prompt(row, tokenizer)
                embeds, labels, mask = build_inputs(model, tokenizer, prompt_ids, row["answer"], latents)
                # Avoid full CausalLM logits [sequence, 152k] (which can cost
                # >9 GiB in float32). Only the answer-token positions are
                # supervised, so run the frozen backbone and project those
                # final hidden states through lm_head.
                backbone_outputs = model.model(
                    inputs_embeds=embeds,
                    attention_mask=mask,
                    use_cache=False,
                    return_dict=True,
                )
                hidden = backbone_outputs.last_hidden_state
                require_finite(
                    hidden,
                    "hidden",
                    args,
                    adbt,
                    optimizer,
                    epoch,
                    current_video_index,
                    global_step,
                    current_video,
                    tensors={"latent": latents, "inputs_embeds": embeds},
                )
                # Causal LM alignment: the hidden state at position t predicts
                # the token at t+1.  The answer label is therefore selected
                # from labels[:, 1:] while its predictor is hidden[:, :-1].
                shifted_labels = labels[:, 1:].to(hidden.device)
                shifted_hidden = hidden[:, :-1]
                answer_mask = shifted_labels.ne(-100)
                answer_hidden = shifted_hidden[answer_mask]
                answer_labels = shifted_labels[answer_mask]
                lm_head = model.lm_head
                answer_logits = lm_head(answer_hidden)
                require_finite(
                    answer_logits,
                    "answer_logits",
                    args,
                    adbt,
                    optimizer,
                    epoch,
                    current_video_index,
                    global_step,
                    current_video,
                    tensors={"latent": latents, "hidden": hidden},
                )
                raw_loss = F.cross_entropy(answer_logits.float(), answer_labels)
                loss = raw_loss / args.gradient_accumulation_steps
                require_finite(
                    loss,
                    "loss",
                    args,
                    adbt,
                    optimizer,
                    epoch,
                    current_video_index,
                    global_step,
                    current_video,
                    tensors={"latent": latents, "answer_logits": answer_logits},
                )
                loss.backward()
                raw_loss_value = float(raw_loss.detach().cpu())
                video_loss_sum += raw_loss_value
                video_loss_count += 1
                if global_step == 0:
                    grad_norm = sum(
                        float(parameter.grad.detach().float().norm().cpu())
                        for parameter in adbt.parameters() if parameter.grad is not None
                    )
                    if not math.isfinite(grad_norm) or grad_norm == 0.0:
                        raise RuntimeError(f"ADBT received no finite gradient (norm={grad_norm})")
                    LOGGER.info("first-step ADBT gradient sum-norm=%.9e", grad_norm)
                clipped_grad_norm = None
                if (global_step + 1) % args.gradient_accumulation_steps == 0:
                    try:
                        clipped_grad_norm = float(
                            torch.nn.utils.clip_grad_norm_(
                                adbt.parameters(),
                                args.max_grad_norm,
                                error_if_nonfinite=True,
                            )
                        )
                    except RuntimeError:
                        save_debug_state(
                            args,
                            adbt,
                            optimizer,
                            epoch,
                            current_video_index,
                            global_step,
                            current_video,
                            reason="nonfinite_gradient_norm",
                            tensors={"latent": latents, "answer_logits": answer_logits},
                        )
                        raise
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if wandb_run is not None and global_step % args.wandb_log_every == 0:
                    metrics = {
                        "train/step": global_step,
                        "train/loss": raw_loss_value,
                        "train/loss_scaled": float(loss.detach().cpu()),
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "train/epoch": epoch + 1,
                        "train/video_index": video_offset + video_idx + 1,
                        "train/video_progress": (video_offset + video_idx + 1) / max(1, len(video_order)),
                        "train/answer_tokens": int(answer_labels.numel()),
                        "adbt_latent/max": latent_stats["max"],
                        "adbt_latent/min": latent_stats["min"],
                        "adbt_latent/mean": latent_stats["mean"],
                        "adbt_latent/std": latent_stats["std"],
                        "adbt_latent/max_abs": latent_stats["max_abs"],
                        "adbt_latent/nonfinite": latent_stats["nonfinite"],
                        "adbt_attention/entropy": adbt_stats["attention_entropy_mean"],
                        "adbt_attention/effective_support": adbt_stats["effective_support_mean"],
                        "adbt_attention/unique_argmax": adbt_stats["unique_argmax_mean"],
                        "adbt_attention/top1_minus_top2": adbt_stats["top1_minus_top2_mean"],
                        "adbt_query/mean_offdiag_cosine": adbt_stats["query_cosine_mean"],
                        "adbt_query/max_cosine": adbt_stats["query_cosine_max"],
                        "adbt_qk/q_norm_mean": adbt_stats["q_norm_mean"],
                        "adbt_qk/q_norm_max": adbt_stats["q_norm_max"],
                        "adbt_qk/k_norm_mean": adbt_stats["k_norm_mean"],
                        "adbt_qk/k_norm_max": adbt_stats["k_norm_max"],
                        "adbt_qk/score_min": adbt_stats["score_min"],
                        "adbt_qk/score_max": adbt_stats["score_max"],
                        "adbt_qk/score_std": adbt_stats["score_std"],
                    }
                    if clipped_grad_norm is not None:
                        metrics["train/grad_norm"] = clipped_grad_norm
                    wandb_run.log(metrics, step=global_step)
            if global_step % args.log_every == 0:
                mean_video_loss = video_loss_sum / max(1, video_loss_count)
                LOGGER.info(
                    "epoch=%d video=%d/%d step=%d loss=%.5f entropy=%.4f "
                    "support=%.3f qcos=%.6f unique=%.2f gap=%.4f",
                    epoch,
                    video_idx + 1,
                    len(video_order),
                    global_step,
                    mean_video_loss,
                    adbt_stats["attention_entropy_mean"],
                    adbt_stats["effective_support_mean"],
                    adbt_stats["query_cosine_mean"],
                    adbt_stats["unique_argmax_mean"],
                    adbt_stats["top1_minus_top2_mean"],
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/step": global_step,
                            "train/video_loss": mean_video_loss,
                            "train/video_frames": int(frames.shape[0]),
                            "train/video_duration_seconds": float(timestamps[-1]) if len(timestamps) else 0.0,
                        },
                        step=global_step,
                    )
            # Evaluate only after the current video's features are released.
            # A video has three questions, so the boundary may be crossed by a
            # few steps; the metric is tagged with the actual completed step.
            if next_eval_step is not None and global_step >= next_eval_step:
                del (
                    visual,
                    audio,
                    frames,
                    timestamps,
                    latents,
                    embeds,
                    labels,
                    mask,
                    backbone_outputs,
                    hidden,
                    shifted_labels,
                    shifted_hidden,
                    answer_mask,
                    answer_hidden,
                    answer_labels,
                    answer_logits,
                    loss,
                    raw_loss,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                eval_metrics = evaluate_small_set(
                    model,
                    tokenizer,
                    image_processor,
                    adbt,
                    audio_encoder,
                    dataset,
                    split,
                    max_videos=args.eval_max_videos,
                    max_frames=args.eval_max_frames,
                    vision_batch_size=args.vision_batch_size,
                    video_offset=(global_step // max(1, args.eval_every_steps) - 1)
                    * max(1, args.eval_max_videos),
                )
                LOGGER.info(
                    "heldout eval step=%d videos=%d examples=%d loss=%.5f first_token_acc=%.4f exact_acc=%.4f",
                    global_step,
                    eval_metrics["videos"],
                    eval_metrics["examples"],
                    eval_metrics["loss"],
                    eval_metrics["first_token_accuracy"],
                    eval_metrics["exact_match_accuracy"],
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/step": global_step,
                            "eval/loss": eval_metrics["loss"],
                            "eval/first_token_accuracy": eval_metrics["first_token_accuracy"],
                            "eval/exact_match_accuracy": eval_metrics["exact_match_accuracy"],
                            "eval/examples": eval_metrics["examples"],
                            "eval/videos": eval_metrics["videos"],
                            "eval/frames": eval_metrics["frames"],
                        },
                        step=global_step,
                    )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                while next_eval_step <= global_step:
                    next_eval_step += args.eval_every_steps
            overall_video_idx = video_offset + video_idx + 1
            if overall_video_idx % args.save_every_videos == 0:
                progress = Path(args.output_dir) / f"adbt_epoch_{epoch + 1}_video_{overall_video_idx}.pt"
                progress.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "audio_bottleneck": adbt.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "video_idx": overall_video_idx,
                    "global_step": global_step,
                    "args": vars(args),
                    "init_adapter": args.init_adapter,
                    "split_file": args.split_file,
                }, progress)
                LOGGER.info("saved progress checkpoint %s", progress)
        out = Path(args.output_dir) / f"adbt_epoch_{epoch + 1}.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "audio_bottleneck": adbt.state_dict(),
            "epoch": epoch,
            "video_idx": len(video_order) + video_offset,
            "global_step": global_step,
            "args": vars(args),
            "init_adapter": args.init_adapter,
            "split_file": args.split_file,
        }, out)
        LOGGER.info("saved %s", out)
    if wandb_run is not None:
        wandb_run.summary["final_global_step"] = global_step
        wandb_run.summary["completed_epochs"] = args.epochs
        wandb_run.finish()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", default="lmms-lab/llava-onevision-qwen2-7b-ov")
    parser.add_argument("--hf-cache", default="/home/yxd/.cache/huggingface")
    parser.add_argument("--beats-checkpoint", default="/nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt")
    parser.add_argument("--split-file", default="./results/adbt_videomme_split.json")
    parser.add_argument("--output-dir", default="./results/adbt-checkpoints")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--train-max-frames", type=int, default=32)
    parser.add_argument("--num-queries", type=int, default=32)
    parser.add_argument("--bottleneck-hidden", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument(
        "--architecture",
        choices=("legacy", "phase4"),
        default="phase4",
        help="Audio-query/attention architecture version.",
    )
    parser.add_argument(
        "--attention-temperature",
        type=float,
        default=0.2,
        help="Fixed cosine-attention temperature used by Phase 4.",
    )
    parser.add_argument(
        "--value-mode",
        choices=("projected", "native"),
        default="native",
        help=(
            "projected uses the legacy learned Value/output path; native uses "
            "the Phase-3 M=softmax(QK^T)V_original path exactly."
        ),
    )
    parser.add_argument(
        "--latent-norm",
        choices=("rmsnorm", "layernorm", "none"),
        default=None,
        help=(
            "Normalization applied to projected-value latents. Defaults to rmsnorm "
            "for projected mode and none for native mode; native requires none."
        ),
    )
    parser.add_argument("--vision-batch-size", type=int, default=16)
    parser.add_argument("--beats-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-every-videos", type=int, default=20)
    parser.add_argument(
        "--eval-every-steps",
        type=int,
        default=100,
        help="Run the small held-out diagnostic every N completed question steps; 0 disables it.",
    )
    parser.add_argument(
        "--eval-max-videos",
        type=int,
        default=4,
        help="Number of held-out videos used by the periodic diagnostic; 0 uses all test videos.",
    )
    parser.add_argument(
        "--eval-max-frames",
        type=int,
        default=32,
        help="Maximum frames per held-out video for the periodic diagnostic.",
    )
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--resume",
        default="",
        help="Resume weights, optimizer, epoch, video index, and global step from a progress checkpoint.",
    )
    checkpoint_group.add_argument(
        "--init-adapter",
        default="",
        help=(
            "Initialize adapter weights without restoring training progress. If only --num-queries "
            "changes, existing query slots are copied and new slots retain independent initialization."
        ),
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("WANDB_TRAIN", "0").lower() in {"1", "true", "yes", "on"},
        help="Log training metrics to Weights & Biases.",
    )
    parser.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "streamingtom-videomme"))
    parser.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY", ""))
    parser.add_argument("--wandb-run-name", default=os.getenv("WANDB_NAME", ""))
    parser.add_argument("--wandb-run-id", default=os.getenv("WANDB_RUN_ID", ""))
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.getenv("WANDB_MODE", "online"),
        help="W&B transport mode; use offline on a compute node without network access.",
    )
    parser.add_argument(
        "--wandb-log-every",
        type=int,
        default=1,
        help="Log per-question metrics every N optimizer steps (video summaries use --log-every).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    train(parse_args())
