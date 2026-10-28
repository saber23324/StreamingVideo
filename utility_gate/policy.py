"""Default-off hard selection and immutable, chronological segment routing."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .features import GateFeatures, prefix_features, take_features


@dataclass(frozen=True)
class UtilityDecision:
    utility_score: torch.Tensor
    accepted: torch.Tensor
    valid_audio: torch.Tensor
    threshold: float


class UtilityRoutingPolicy(nn.Module):
    def __init__(self, audio_router, vision_router=None, gate=None, threshold=0., enabled=False):
        super().__init__()
        self.audio_router, self.vision_router, self.gate = audio_router, vision_router, gate
        self.enabled, self.threshold, self.last_decision = bool(enabled), float(threshold), None
        if not math.isfinite(self.threshold):
            raise ValueError('finite frozen validation threshold required')
        if self.enabled:
            if vision_router is None or gate is None:
                raise ValueError('enabled utility policy requires an explicit vision checkpoint and gate')
            for module in (audio_router,vision_router,gate):
                if module.training or any(p.requires_grad for p in module.parameters()):
                    raise ValueError('deployment gate and baseline checkpoints must be frozen/eval')
            if audio_router.stage != 3 or vision_router.stage != 1 or audio_router.num_queries != vision_router.num_queries:
                raise ValueError('expected matched-budget stage-3/stage-1 baseline interfaces')

    @torch.inference_mode()
    def choose(self, features):
        if not self.enabled:
            raise RuntimeError('disabled policy directly delegates to the legacy router')
        prediction = self.gate.predict(features)
        accepted = prediction.valid_audio & torch.isfinite(prediction.utility_score) & (prediction.utility_score > self.threshold)
        self.last_decision = UtilityDecision(prediction.utility_score,accepted,prediction.valid_audio,self.threshold)
        return self.last_decision

    @torch.inference_mode()
    def resample(self, visual, audio, timestamps, decision):
        if decision.accepted.shape != (1,):
            raise ValueError('prefix policy expects one whole-prefix choice')
        if bool(decision.accepted[0]) and audio is not None:
            return self.audio_router(visual,audio,timestamps)
        return self.vision_router(visual,None,timestamps)

    def forward(self, visual, audio=None, timestamps=None, features=None):
        if not self.enabled:
            # Absolutely no pooling, casting or masked interpolation on the legacy path.
            return self.audio_router(visual,audio,timestamps)
        if features is None:
            raise ValueError('gold/query-free GateFeatures required')
        if audio is None:
            features = GateFeatures(features.visual,features.audio,features.context,torch.zeros_like(features.valid_audio))
        return self.resample(visual,audio,timestamps,self.choose(features))


class CausalStreamRouter:
    """Append one decision's visual tokens; no later step can revise the bank."""
    def __init__(self, policy):
        if not policy.enabled:
            raise ValueError('streaming gate requires an enabled frozen policy')
        self.policy = policy
        self.visual_means, self.audio_means, self.valid, self.timestamps = [], [], [], []
        self.latents, self.trace = [], []

    @torch.inference_mode()
    def step(self, visual, audio, timestamp, valid_audio, frame_pts=None):
        timestamp = float(timestamp)
        if not math.isfinite(timestamp) or timestamp < 0 or (self.timestamps and timestamp <= self.timestamps[-1]):
            raise ValueError('strictly increasing finite decision timestamps required')
        if frame_pts is not None and frame_pts > timestamp+1e-6:
            raise ValueError('future frame')
        if visual.ndim != 2 or visual.shape[0] != 196 or not torch.isfinite(visual).all():
            raise ValueError('finite current native visual tokens required')
        okay = bool(valid_audio) and audio is not None and bool(torch.isfinite(audio).all())
        current_audio = audio.detach().float() if okay else torch.zeros((1,768),device=visual.device)
        self.visual_means.append(visual.detach().float().mean(0).clone())
        self.audio_means.append(current_audio.mean(0).clone())
        self.valid.append(okay)
        previous = self.timestamps[-1] if self.timestamps else None
        self.timestamps.append(timestamp)
        history = prefix_features(torch.stack(self.visual_means)[:,None],torch.stack(self.audio_means)[:,None],
                                  self.valid,self.timestamps)
        current = take_features(history,slice(-1,None))
        current = GateFeatures(current.visual,current.audio,current.context,
                               torch.tensor([okay],device=visual.device,dtype=torch.bool))
        decision = self.policy.choose(current)
        if bool(decision.accepted[0]):
            if previous is None:
                z,p = self.policy.audio_router(visual[None].float(),current_audio[None],
                                               torch.tensor([timestamp],device=visual.device))
            else:
                # A duplicate of CURRENT inputs supplies the previous timestamp
                # for the original delta-time implementation. No historical V
                # is re-routed and only the final independent batch row is kept.
                z,p = self.policy.audio_router(visual[None].float().expand(2,-1,-1),
                    current_audio[None].expand(2,-1,-1),torch.tensor([previous,timestamp],device=visual.device))
                z,p = z[-1:],p[-1:]
        else:
            z,p = self.policy.vision_router(visual[None].float(),None,
                                           torch.tensor([timestamp],device=visual.device))
        self.latents.append(z[0].detach().clone())
        self.trace.append({'timestamp_s':timestamp,'frame_pts_s':frame_pts,'valid_audio':okay,
            'audio_window_start_s':max(0.,timestamp-2.),'audio_window_end_s':timestamp,
            'accepted':bool(decision.accepted[0]),'utility_score':float(decision.utility_score[0]),
            'threshold':decision.threshold,'visual_tokens':int(z.shape[1]),'historical_tokens_revised':False})
        return z[0],p[0]

    def token_bank(self):
        if not self.latents:
            raise ValueError('no observations yet')
        return torch.stack(self.latents)
