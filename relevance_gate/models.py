"""Frozen-feature global / prototype gates, independent of the VLM."""

import torch
from torch import nn
from torch.nn import functional as F

from .types import RelevanceGateOutput


def masked_pool(tokens, mask=None):
    """Ignore padding NaNs; reject a row with nonfinite unmasked features."""
    if tokens.ndim == 2:
        tokens = tokens[:, None, :]
    if tokens.ndim != 3 or tokens.shape[-1] == 0:
        raise ValueError("features must be [B,L,D] or [B,D]")
    if mask is None:
        mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
    if mask.dtype != torch.bool or mask.shape != tokens.shape[:2]:
        raise ValueError("mask must be boolean [B,L]")
    finite = torch.isfinite(tokens).all(-1)
    valid = mask.any(-1) & (finite | ~mask).all(-1)
    clean = torch.where((mask & finite)[..., None], tokens.detach().float(), 0.)
    pooled = clean.sum(1) / mask.sum(1).clamp_min(1)[..., None]
    return pooled, valid


class PooledAVGate(nn.Module):
    def __init__(self, visual_dim, audio_dim=768, hidden=128, dropout=0.1, variant="g1"):
        super().__init__()
        if variant not in ("g1", "g2"):
            raise ValueError("variant must be g1 or g2")
        self.spec = dict(visual_dim=visual_dim, audio_dim=audio_dim, hidden=hidden,
                         dropout=dropout, variant=variant)
        self.visual_projection = nn.Linear(visual_dim, hidden)
        self.audio_projection = nn.Linear(audio_dim, hidden)
        self.head = nn.Sequential(nn.Linear(hidden * (4 if variant == "g1" else 2), hidden),
                                  nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, visual, audio, visual_mask=None, audio_mask=None):
        v, vv = masked_pool(visual, visual_mask)
        a, av = masked_pool(audio, audio_mask)
        if v.shape[0] != a.shape[0]:
            raise ValueError("audio and visual batch sizes differ")
        v, a = self.visual_projection(v), self.audio_projection(a)
        features = [v, a, v * a, (v - a).abs()] if self.spec["variant"] == "g1" else [v, a]
        return self.head(torch.cat(features, -1)).squeeze(-1), vv & av, None


class PrototypeAVGate(nn.Module):
    """G3 prototype matcher with an independent match-vs-null classifier.

    Local attention is conditional evidence allocation, never match confidence.
    Local outputs require supervised source masks during training.
    """
    def __init__(self, visual_dim, audio_dim=768, hidden=128, dropout=0.1, prototypes=8):
        super().__init__()
        self.spec = dict(visual_dim=visual_dim, audio_dim=audio_dim, hidden=hidden,
                         dropout=dropout, prototypes=prototypes)
        self.visual_projection = nn.Linear(visual_dim, hidden)
        self.audio_projection = nn.Linear(audio_dim, hidden)
        self.prototype_queries = nn.Parameter(torch.randn(prototypes, hidden) * 0.02)
        self.match_null_head = nn.Sequential(nn.Linear(hidden * 3 + 2, hidden),
                                             nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, visual, audio, visual_mask=None, audio_mask=None):
        if visual.ndim != 3 or visual.shape[1] == 0:
            raise ValueError("prototype matching requires nonempty visual [B,N,D]")
        vmean, vv = masked_pool(visual, visual_mask)
        amean, av = masked_pool(audio, audio_mask)
        if visual_mask is None:
            visual_mask = torch.ones(visual.shape[:2], dtype=torch.bool, device=visual.device)
        detached = visual.detach().float()
        v = self.visual_projection(torch.where(torch.isfinite(detached), detached, 0.))
        a = self.audio_projection(amean)
        # All-invalid rows receive a temporary dummy position, then are rejected.
        safe_mask = visual_mask.clone()
        safe_mask[:, 0] |= ~safe_mask.any(-1)
        assignments = torch.einsum("kh,bnh->bkn", self.prototype_queries, v)
        assignments = assignments.masked_fill(~safe_mask[:, None], -torch.inf).softmax(-1)
        prototypes = assignments @ v
        similarity = torch.einsum("bkh,bh->bk", F.normalize(prototypes, dim=-1), F.normalize(a, dim=-1))
        weights = (similarity / 0.1).softmax(-1)
        local = torch.einsum("bk,bkn->bn", weights, assignments)
        features = torch.cat([self.visual_projection(vmean), a,
                              (weights[..., None] * prototypes).sum(1),
                              similarity.max(-1).values[:, None],
                              torch.logsumexp(similarity, -1)[:, None]], -1)
        binary_logits = self.match_null_head(features)
        return binary_logits[:, 1] - binary_logits[:, 0], vv & av, local


class ImageBindCosineGate(nn.Module):
    """G0 expects embeddings from the SAME frozen ImageBind checkpoint."""
    def forward(self, visual, audio, visual_mask=None, audio_mask=None):
        v, vv = masked_pool(visual, visual_mask)
        a, av = masked_pool(audio, audio_mask)
        if v.shape != a.shape:
            raise ValueError("ImageBind audio/image embeddings must share shape and space")
        valid = vv & av & (v.norm(dim=-1) > 0) & (a.norm(dim=-1) > 0)
        cosine = (F.normalize(v, dim=-1) * F.normalize(a, dim=-1)).sum(-1).clamp(-1, 1)
        return cosine, valid, None


def predict(model, visual, audio, valid_aux, calibration, visual_mask=None, audio_mask=None):
    if calibration is None or calibration.get("threshold") is None:
        raise ValueError("a frozen validation calibration and threshold are required")
    raw, valid_features, local = model(visual, audio, visual_mask, audio_mask)
    if valid_aux.shape != raw.shape or valid_aux.dtype != torch.bool:
        raise ValueError("valid_aux must be boolean [B]")
    valid = valid_aux & valid_features & torch.isfinite(raw)
    logit = torch.nan_to_num(raw.float())
    probability = torch.sigmoid(logit * calibration["scale"] + calibration["bias"])
    probability = probability.masked_fill(~valid, 0.)
    accepted = valid & (probability >= calibration["threshold"])
    if local is not None:
        local = torch.nan_to_num(local).masked_fill(~accepted[:, None], 0.)
    cosine = isinstance(model, ImageBindCosineGate)
    reported_logit = logit * calibration["scale"] + calibration["bias"] if cosine else logit
    return RelevanceGateOutput(reported_logit, probability, accepted, valid, local,
                               {"calibration_version": calibration.get("version"),
                                "score_kind": "weak_pairing_calibrated_score" if calibration.get("label_protocol") == "weak_pairing" else "validation_calibrated_probability",
                                "label_protocol": calibration.get("label_protocol", "visible_correspondence"),
                                "raw_score_kind": "cosine_similarity" if cosine else "uncalibrated_model_logit",
                                "match_logit_kind": "validation_calibrated_logit" if cosine else "uncalibrated_model_logit",
                                "raw_score": logit.detach().cpu().tolist()})


def make_model(kind, spec):
    if kind in ("g1", "g2"):
        return PooledAVGate(**{**spec, "variant": kind})
    if kind == "g3":
        return PrototypeAVGate(**spec)
    if kind == "g0":
        return ImageBindCosineGate()
    raise ValueError(f"unknown gate: {kind}")
