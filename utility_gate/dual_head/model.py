"""Tiny shared visual-anchored trunk with distinct correspondence/utility heads."""
from __future__ import annotations

import torch
from torch import nn

from utility_gate.common import sha256
from utility_gate.features import GateFeatures
from utility_gate.gate2_interfaces import load_gate2
from .interfaces import DualHeadOutput, causal_features


class DualHeadGate(nn.Module):
    def __init__(self, visual_dim=3584, audio_dim=768, hidden=64, utility_scale=1., dropout=.1):
        super().__init__()
        self.spec = dict(visual_dim=visual_dim,audio_dim=audio_dim,hidden=hidden,utility_scale=float(utility_scale),dropout=dropout)
        self.visual_projection=nn.Sequential(nn.LayerNorm(visual_dim*2,elementwise_affine=False),nn.Linear(visual_dim*2,hidden),nn.GELU())
        self.audio_projection=nn.Sequential(nn.LayerNorm(audio_dim*2,elementwise_affine=False),nn.Linear(audio_dim*2,hidden),nn.GELU())
        self.comparison=nn.Sequential(nn.Linear(hidden*4+4,hidden),nn.GELU(),nn.Dropout(dropout))
        self.correspondence_head=nn.Linear(hidden,1)
        self.utility_head=nn.Linear(hidden,1)
        self.decision_head=nn.Sequential(nn.Linear(3,8),nn.GELU(),nn.Linear(8,1))
        self.head_status={'correspondence':'untrained','utility':'untrained','decision':'untrained'}
        self.provenance={}

    def forward_features(self, features):
        if not isinstance(features,GateFeatures): raise TypeError('gold/query-free GateFeatures required')
        features.validate()
        if features.visual.shape[-1] != self.spec['visual_dim']*2 or features.audio.shape[-1] != self.spec['audio_dim']*2:
            raise ValueError('expected current and strictly-past feature widths')
        v=self.visual_projection(features.visual.detach());a=self.audio_projection(features.audio.detach())
        h=self.comparison(torch.cat((v,a,v-a,v*a,features.context.detach()),1))
        available=features.valid_audio[:,None]
        corr=torch.where(available,self.correspondence_head(h),torch.zeros_like(available,dtype=h.dtype))
        utility=torch.where(available,self.utility_head(h)*self.spec['utility_scale'],torch.zeros_like(corr))
        result=DualHeadOutput(corr,utility,torch.zeros_like(available),available,
            diagnostics={'head_status':dict(self.head_status),'correspondence_calibrated_probability':False,
                         'selection_default':'off until an admissible validation rule is supplied'})
        result.validate();return result

    def forward(self, inputs, rule=None):
        result=self.forward_features(causal_features(inputs,self.spec['audio_dim']))
        if rule is not None:
            from .calibration import apply_selection
            result=apply_selection(result,rule,self.head_status,self.decision_head)
        return result

    @classmethod
    def from_gate2_utility(cls,path):
        original,payload=load_gate2(path)
        if payload['layout']!='anchor_history': raise ValueError('only the existing causal current/past layout is equivalent')
        spec=original.model.spec
        value=cls(spec['visual_dim']//2,spec['audio_dim']//2,spec['hidden'],spec['target_scale'])
        for name in ('visual_projection','audio_projection','comparison'):
            getattr(value,name).load_state_dict(getattr(original.model,name).state_dict())
        value.utility_head.load_state_dict(original.model.regression.state_dict())
        value.head_status['utility']='frozen_gate2_reference'
        value.provenance={'utility_warmstart_path':str(path),'utility_warmstart_sha256':sha256(path),
                          'correspondence_not_renamed_from_QA_beneficial_head':True}
        return value.eval()


def load_dual_head(path,device='cpu'):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload['kind']!='audiorouter_dual_head': raise ValueError('not a dual-head artifact')
    model=DualHeadGate(**payload['spec'])
    model.load_state_dict(payload['state_dict'],strict=True)
    model.head_status=payload['head_status'];model.provenance=payload['provenance']
    return model.to(device).eval().requires_grad_(False),payload
