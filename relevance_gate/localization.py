"""Mask-to-position mapping for the verified SigLIP 384/14 -> native 14x14 path.

This maps positional token support, not the full receptive field of a global
vision transformer. Ranking metrics here never replace weighted resampling.
"""
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .manifest import file_sha256


LAYOUT = {"version": "siglip384_patch14_valid_bilinear27to14_v1", "input_size": 384,
          "patch_size": 14, "encoder_grid": 27, "native_grid": 14, "align_corners": False,
          "mask_resize": "nearest", "support": "positional patch support, not a transformer's full receptive field"}


def validate_layout(metadata):
    processor = metadata.get("processor", {})
    if metadata.get("kind") != "native" or metadata.get("pretrained") != "lmms-lab/llava-onevision-qwen2-7b-ov" or processor.get("class") != "SigLipImageProcessor" or processor.get("size") != [384, 384] or processor.get("applied_crop") is not False:
        raise ValueError("localization mapping requires verified SigLIP 384 resize without crop")


def native_mask_target(mask, metadata):
    validate_layout(metadata)
    mask = torch.as_tensor(mask)
    if mask.ndim != 2 or not torch.isfinite(mask).all():
        raise ValueError("finite 2D source mask required")
    resized = F.interpolate((mask > 0).float()[None, None], size=(384, 384), mode="nearest")
    patches = F.avg_pool2d(resized, kernel_size=14, stride=14)
    if patches.shape[-2:] != (27, 27):
        raise AssertionError("unexpected valid-convolution patch grid")
    native = F.interpolate(patches, size=(14, 14), mode="bilinear", align_corners=False).flatten()
    if native.sum() <= 0:
        raise ValueError("source has no support in the consumed native patch positions")
    return native


def attach_local_target(row, metadata):
    result = dict(row)
    if row["visible_correspondence"] == 1:
        if not row.get("source_region_mask") or not row.get("frame_path"):
            raise ValueError("positive pair has no defensible source-region mask and verified frame for localization")
        path = Path(row["source_region_mask"])
        with Image.open(path) as image:
            mask = np.asarray(image).copy()
            mask_size = image.size
        with Image.open(row["frame_path"]) as image:
            if image.size != mask_size:
                raise ValueError("source mask and verified frame geometry differ")
        target = native_mask_target(mask, metadata)
        result.update(local_token_target=target.tolist(), local_target_provenance={**LAYOUT, "source_mask_sha256": file_sha256(path)})
    return result


def validate_local_target(row, metadata):
    validate_layout(metadata)
    evidence = row.get("local_target_provenance", {})
    if evidence.get("version") != LAYOUT["version"] or not row.get("source_region_mask"):
        raise ValueError("G3 target needs verified native layout and source-mask provenance")
    if file_sha256(row["source_region_mask"]) != evidence.get("source_mask_sha256"):
        raise ValueError("localization source mask changed")
    target = torch.tensor(row["local_token_target"])
    if target.shape != (196,) or not torch.isfinite(target).all() or (target < 0).any() or target.sum() <= 0:
        raise ValueError("invalid native localization target")


def localization_metrics(records, budget=64):
    positives = [r for r in records if r.get("semantic_visible_correspondence", r["visible_correspondence"]) == 1
                 and r.get("valid_aux") and r.get("local_token_target") and r.get("conditional_local_scores")]
    recalls, masses = [], []
    for row in positives:
        scores, target = torch.tensor(row["conditional_local_scores"]), torch.tensor(row["local_token_target"])
        selected = scores.topk(min(budget, len(scores))).indices
        recalls.append(float(target[selected].sum() / target.sum()))
        masses.append(float(scores[target > 0].sum()))
    return {"official_positive_pairs": len(positives), "ranking_budget": budget,
            "area_weighted_token_recall": float(np.mean(recalls)) if recalls else None,
            "conditional_mass_on_source_support": float(np.mean(masses)) if masses else None,
            "scope": "conditional localization on official positives; ranking diagnostic only, no hard-pruning or global correspondence claim"}
