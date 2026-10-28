"""Gold/query-free deployable feature schema and causal prefix summaries."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class GateFeatures:
    visual: torch.Tensor         # [B, native visual width]
    audio: torch.Tensor          # [B, BEATs width]
    context: torch.Tensor        # [B, 4] timestamp/delta/coverage/history length
    valid_audio: torch.Tensor    # [B] bool

    def to(self, device):
        return GateFeatures(*(value.to(device) for value in (self.visual, self.audio, self.context, self.valid_audio)))

    def validate(self):
        if self.visual.ndim != 2 or self.audio.ndim != 2 or self.context.shape != (len(self.visual), 4):
            raise ValueError('expected batched pooled visual/audio and four causal context values')
        if len(self.audio) != len(self.visual) or self.valid_audio.shape != (len(self.visual),):
            raise ValueError('batch size mismatch')
        if self.valid_audio.dtype != torch.bool:
            raise ValueError('valid_audio must be boolean')
        if not all(torch.isfinite(x).all() for x in (self.visual, self.audio, self.context)):
            raise ValueError('nonfinite gate input')


def prefix_features(visual, audio, valid, timestamps):
    """Return one summary for each prefix; no question/answer can enter here."""
    if visual.ndim != 3 or audio.ndim != 3 or len(visual) != len(audio):
        raise ValueError('expected frame/token/channel tensors')
    timestamps = torch.as_tensor(timestamps, device=visual.device, dtype=torch.float32)
    valid = torch.as_tensor(valid, device=visual.device, dtype=torch.bool)
    if timestamps.shape != (len(visual),) or valid.shape != timestamps.shape or len(visual) == 0:
        raise ValueError('timestamp/availability shape mismatch')
    if not torch.isfinite(timestamps).all() or (timestamps < 0).any() or (timestamps[1:] < timestamps[:-1]).any():
        raise ValueError('timestamps must be finite, nonnegative and sorted')
    v = visual.detach().float().mean(1)
    a = audio.detach().float().mean(1)
    if not torch.isfinite(v).all():
        raise ValueError('nonfinite visual features')
    valid = valid & torch.isfinite(a).all(1)
    a = torch.where(valid[:, None], a, torch.zeros_like(a))
    count = torch.arange(1, len(v)+1, device=v.device, dtype=torch.float32)
    acount = valid.float().cumsum(0)
    v = v.cumsum(0) / count[:, None]
    a = a.cumsum(0) / acount.clamp_min(1)[:, None]
    dt = torch.cat((timestamps[:1] * 0, timestamps[1:] - timestamps[:-1]))
    context = torch.stack((torch.log1p(timestamps), torch.log1p(dt), acount/count, torch.log1p(count)), 1)
    result = GateFeatures(v, a, context, acount > 0)
    result.validate()
    return result


def take_features(features, positions):
    return GateFeatures(*(getattr(features, key)[positions] for key in ('visual', 'audio', 'context', 'valid_audio')))


def feature_payload(features):
    features.validate()
    return {key: getattr(features, key).detach().cpu() for key in ('visual', 'audio', 'context', 'valid_audio')}


def load_feature_payload(payload):
    if set(payload) != {'visual', 'audio', 'context', 'valid_audio'}:
        raise ValueError('unexpected feature field; QA labels and identities are forbidden in inference tensors')
    value = GateFeatures(**payload)
    value.validate()
    return value
