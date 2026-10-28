"""Tiny utility regression with separate projections of unaligned modalities."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .features import GateFeatures


@dataclass(frozen=True)
class UtilityPrediction:
    utility_score: torch.Tensor
    beneficial_logit: torch.Tensor
    valid_audio: torch.Tensor


class TinyUtilityGate(nn.Module):
    def __init__(self, visual_dim=3584, audio_dim=768, hidden=64, target_scale=1.):
        super().__init__()
        self.spec = {'visual_dim': visual_dim, 'audio_dim': audio_dim, 'hidden': hidden, 'target_scale': float(target_scale)}
        self.visual_projection = nn.Sequential(nn.LayerNorm(visual_dim, elementwise_affine=False),
                                               nn.Linear(visual_dim, hidden), nn.GELU())
        self.audio_projection = nn.Sequential(nn.LayerNorm(audio_dim, elementwise_affine=False),
                                              nn.Linear(audio_dim, hidden), nn.GELU())
        self.comparison = nn.Sequential(nn.Linear(hidden*4+4, hidden), nn.GELU(), nn.Dropout(.1))
        self.regression = nn.Linear(hidden, 1)
        self.beneficial = nn.Linear(hidden, 1)

    def forward(self, features: GateFeatures):
        if not isinstance(features, GateFeatures):
            raise TypeError('only the gold/query-free GateFeatures schema is accepted')
        features.validate()
        v = self.visual_projection(features.visual.detach())
        a = self.audio_projection(features.audio.detach())
        h = self.comparison(torch.cat((v, a, v-a, v*a, features.context.detach()), 1))
        return UtilityPrediction(self.regression(h).squeeze(1)*self.spec['target_scale'],
                                 self.beneficial(h).squeeze(1), features.valid_audio)

    @torch.inference_mode()
    def predict(self, visual_features: GateFeatures):
        if self.training:
            raise RuntimeError('deployment prediction requires eval mode')
        return self(visual_features)


def load_gate(path, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['kind'] != 'tiny_utility_mlp':
        raise ValueError('not a Utility Gate checkpoint')
    model = TinyUtilityGate(**payload['spec']).to(device)
    model.load_state_dict(payload['state_dict'], strict=True)
    return model.eval().requires_grad_(False), payload
