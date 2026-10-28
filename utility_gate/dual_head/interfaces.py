"""Question/gold-free causal inputs and independent modality output slots."""
from dataclasses import dataclass, field

import torch

from utility_gate.features import GateFeatures, take_features
from utility_gate.gate2_interfaces import AuxiliaryStream, visual_anchored_features


@dataclass(frozen=True)
class DualHeadInput:
    visual_tokens: torch.Tensor          # [B,T,N,Dv], native values
    observation_times: torch.Tensor      # [B,T], scheduled causal arrivals
    decision_times: torch.Tensor         # [B]
    audio: AuxiliaryStream | None = None
    frame_pts: torch.Tensor | None = None
    visual_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class DualHeadOutput:
    correspondence_logit: torch.Tensor   # [B,M], never called calibrated probability
    utility_delta: torch.Tensor          # [B,M], signed regression output
    selection_mask: torch.Tensor         # [B,M] bool
    availability_mask: torch.Tensor      # [B,M] bool, engineering validity only
    modality_names: tuple[str, ...] = ('audio',)
    diagnostics: dict = field(default_factory=dict)

    def validate(self):
        shape = self.utility_delta.shape
        if len(shape) != 2 or shape[-1] != len(self.modality_names): raise ValueError('expected[B,M] independent outputs')
        if any(x.shape != shape for x in (self.correspondence_logit, self.selection_mask, self.availability_mask)):
            raise ValueError('output shapes differ')
        if self.selection_mask.dtype != torch.bool or self.availability_mask.dtype != torch.bool:
            raise ValueError('explicit boolean output masks required')
        if not torch.isfinite(self.utility_delta).all() or not torch.isfinite(self.correspondence_logit).all():
            raise ValueError('nonfinite dual-head outputs')
        if (self.selection_mask & ~self.availability_mask).any(): raise ValueError('unavailable modality selected')


def causal_features(value, audio_dim=768):
    if not isinstance(value, DualHeadInput): raise TypeError('only DualHeadInput is accepted; no question/gold/source dicts')
    visual, times = value.visual_tokens, value.observation_times
    if visual.ndim != 4 or times.shape != visual.shape[:2] or value.decision_times.shape != (len(visual),):
        raise ValueError('explicit[B,T,N,D] visual, [B,T] clock and[B] decisions required')
    if not torch.isfinite(value.decision_times).all(): raise ValueError('invalid decision clock')
    mask = torch.ones_like(times, dtype=torch.bool) if value.visual_mask is None else value.visual_mask
    if mask.shape != times.shape or mask.dtype != torch.bool: raise ValueError('invalid visual mask')
    pts = times if value.frame_pts is None else value.frame_pts
    if pts.shape != times.shape: raise ValueError('invalid visual PTS layout')
    audio = value.audio
    if audio is not None and (audio.features.shape[:2] != times.shape or audio.features.shape[-1] != audio_dim):
        raise ValueError('audio observations must share frame/arrival indexing')
    outputs = []
    for b in range(len(visual)):
        allowed = mask[b] & torch.isfinite(times[b]) & torch.isfinite(pts[b])
        allowed &= (times[b] <= value.decision_times[b]) & (pts[b] <= times[b])
        index = torch.where(allowed)[0]
        if not len(index) or not torch.isclose(times[b,index[-1]], value.decision_times[b], atol=1e-6, rtol=0):
            raise ValueError('no causal current visual observation')
        selected_time = times[b,index][None]
        if (selected_time[:,1:] <= selected_time[:,:-1]).any(): raise ValueError('non-increasing observed times')
        # Future visual/audio states are removed BEFORE pooling/validation, so
        # arbitrary future values cannot perturb the current two-head outputs.
        current = visual[b,index][None]
        if audio is None:
            stream = AuxiliaryStream(torch.zeros(1,len(index),1,audio_dim,device=visual.device),
                selected_time[:,:,None],torch.zeros(1,len(index),device=visual.device,dtype=torch.bool),'audio')
        else:
            stream = AuxiliaryStream(audio.features[b,index][None], audio.timestamps[b,index][None],
                audio.available[b,index][None], audio.modality_name,
                audio.token_mask[b,index][None] if audio.token_mask is not None else None)
        feature = visual_anchored_features(current, selected_time, stream)
        outputs.append(take_features(feature, slice(-1,None)))
    return GateFeatures(*(torch.cat([getattr(v,k) for v in outputs]) for k in ('visual','audio','context','valid_audio')))
