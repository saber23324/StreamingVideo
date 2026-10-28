"""Visual-anchored, causal, independently masked modality selection contracts.

Only audio has a production encoder/model in this phase. Multi-column mask
tests are interface tests, never evidence of real IMU/GUI experiments.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .features import GateFeatures,prefix_features
from .models import TinyUtilityGate, UtilityPrediction


@dataclass(frozen=True)
class AuxiliaryStream:
    features: torch.Tensor              # [B,T,L,D], encoder states
    timestamps: torch.Tensor            # [B,T,L], latest observation used by each state
    available: torch.Tensor             # [B,T] bool
    modality_name: str
    token_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class ModalityGateOutput:
    gates: torch.Tensor                 # [B,T,M], independent sigmoids, zero if absent
    logits: torch.Tensor                # [B,T,M], utility-regression units, not a match probability
    availability: torch.Tensor          # [B,T,M] bool
    modality_names: tuple[str,...]

    def hard_selection(self,threshold):
        return self.availability & (self.logits>threshold)


def independent_gates(logits,availability,modality_names):
    if logits.ndim!=3 or logits.shape!=availability.shape or logits.shape[-1]!=len(modality_names):
        raise ValueError('expected matching [batch,time,modality] output and availability')
    if availability.dtype!=torch.bool or len(set(modality_names))!=len(modality_names):
        raise ValueError('explicit boolean availability and unique modality names required')
    valid=availability & torch.isfinite(logits)
    cleaned=torch.where(valid,logits,torch.zeros_like(logits))
    gates=torch.where(valid,cleaned.sigmoid(),torch.zeros_like(cleaned))
    return ModalityGateOutput(gates,cleaned,valid,tuple(modality_names))


def causal_audio_mask(stream,decision_times):
    if stream.modality_name!='audio':
        raise ValueError('only the real audio frontend is implemented in this phase')
    value=stream.features
    if value.ndim!=4 or stream.timestamps.shape!=value.shape[:-1] or decision_times.shape!=value.shape[:2]:
        raise ValueError('explicit batch/time/audio-state layout required')
    if stream.available.shape!=value.shape[:2] or stream.available.dtype!=torch.bool:
        raise ValueError('audio availability must be boolean [B,T]')
    token_mask=torch.ones_like(stream.timestamps,dtype=torch.bool) if stream.token_mask is None else stream.token_mask
    if token_mask.shape!=stream.timestamps.shape or token_mask.dtype!=torch.bool:
        raise ValueError('invalid audio token mask')
    # Truncate BEFORE pooling. Future/malformed/stale states do not change
    # pooling, history counts or the availability of an otherwise valid prefix.
    allowed=token_mask & torch.isfinite(stream.timestamps)
    allowed &= stream.timestamps<=decision_times[:,:,None]
    allowed &= stream.timestamps>=(decision_times[:,:,None]-2.).clamp_min(0.)
    finite=torch.isfinite(value).all(-1)
    allowed &= finite
    allowed &= stream.available[:,:,None]
    return allowed


def causal_audio_pool(stream,decision_times):
    value=stream.features
    allowed=causal_audio_mask(stream,decision_times)
    cleaned=torch.where(allowed[:,:,:,None],value,torch.zeros_like(value))
    count=allowed.sum(-1)
    pooled=cleaned.sum(-2)/count.clamp_min(1)[:,:,None]
    return pooled,count>0


def assemble_anchor_features(visual_mean,audio_mean,valid_audio,decision_times):
    """Current visual anchor plus strictly previous histories; no QA schema."""
    if visual_mean.ndim!=3 or audio_mean.ndim!=3 or visual_mean.shape[:2]!=audio_mean.shape[:2]:
        raise ValueError('expected [B,T,D] modality summaries')
    if valid_audio.shape!=visual_mean.shape[:2] or decision_times.shape!=valid_audio.shape or valid_audio.dtype!=torch.bool:
        raise ValueError('availability/time shape mismatch')
    if not torch.isfinite(visual_mean).all() or not torch.isfinite(audio_mean).all():
        raise ValueError('pooled features must be finite')
    if not torch.isfinite(decision_times).all() or (decision_times<0).any() or (decision_times[:,1:]<decision_times[:,:-1]).any():
        raise ValueError('finite sorted causal timestamps required')
    visual_mean=visual_mean.detach().float()
    audio_mean=audio_mean.detach().float()
    audio_mean=torch.where(valid_audio[:,:,None],audio_mean,torch.zeros_like(audio_mean))
    batch,time=valid_audio.shape
    previous=torch.arange(time,device=visual_mean.device,dtype=torch.float32)[None,:].expand(batch,-1)
    history_v=(visual_mean.cumsum(1)-visual_mean)/previous.clamp_min(1)[:,:,None]
    available_count=valid_audio.float().cumsum(1)-valid_audio.float()
    history_a=(audio_mean.cumsum(1)-audio_mean)/available_count.clamp_min(1)[:,:,None]
    dt=torch.cat((torch.zeros_like(decision_times[:,:1]),decision_times[:,1:]-decision_times[:,:-1]),1)
    context=torch.stack((torch.log1p(decision_times),torch.log1p(dt),
        available_count/previous.clamp_min(1),torch.log1p(previous+1)),2)
    value=GateFeatures(torch.cat((visual_mean,history_v),2).reshape(batch*time,-1),
        torch.cat((audio_mean,history_a),2).reshape(batch*time,-1),context.reshape(batch*time,4),valid_audio.reshape(-1))
    value.validate()
    return value


def visual_anchored_features(visual,decision_times,stream):
    if visual.ndim!=4 or visual.shape[:2]!=decision_times.shape:
        raise ValueError('visual layout must be [B,T,N,D]')
    pooled,valid=causal_audio_pool(stream,decision_times)
    return assemble_anchor_features(visual.detach().float().mean(-2),pooled,valid,decision_times)


def layout_features(visual,decision_times,stream,layout='anchor_history'):
    if layout=='anchor_history':
        return visual_anchored_features(visual,decision_times,stream)
    if layout!='pooled_history':
        raise ValueError('unknown Gate2 feature-layout ablation')
    audio,valid=causal_audio_pool(stream,decision_times)
    batches=[]
    for index in range(len(visual)):
        value=prefix_features(visual[index],audio[index,:,None],valid[index],decision_times[index])
        batches.append(GateFeatures(value.visual,value.audio,value.context,valid[index]))
    return GateFeatures(*(torch.cat([getattr(v,key) for v in batches]) for key in ('visual','audio','context','valid_audio')))


def features_from_cache(payload,layout='anchor_history'):
    visual=payload['visual'][None]
    audio=payload['causal_audio'][None]
    time=payload['timestamps'][None].float()
    stamp=time[:,:,None].expand(audio.shape[:-1])
    stream=AuxiliaryStream(audio,stamp,payload['audio_valid'][None].bool(),'audio')
    return layout_features(visual,time,stream,layout)


class VisualAnchoredAudioGate(nn.Module):
    def __init__(self,model,layout='anchor_history'):
        super().__init__()
        self.model=model
        self.layout=layout
        self.last_output=None

    def predict(self,features):
        # Compatibility with the existing exact hard-branch policy. This is
        # the only production modality; no sensor encoders are invented.
        raw=self.model.predict(features)
        self.last_output=independent_gates(raw.utility_score[:,None,None],raw.valid_audio[:,None,None],('audio',))
        return UtilityPrediction(self.last_output.logits[:,0,0],raw.beneficial_logit,
                                  self.last_output.availability[:,0,0])

    def forward(self,visual,decision_times,stream):
        batch,time=decision_times.shape
        raw=self.model(layout_features(visual,decision_times,stream,self.layout))
        result=independent_gates(raw.utility_score.reshape(batch,time,1),raw.valid_audio.reshape(batch,time,1),('audio',))
        self.last_output=result
        return result


def load_gate2(path,device='cpu'):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload['kind']!='visual_anchored_audio_gate':
        raise ValueError('expected a Gate2 visual-anchored audio checkpoint')
    tiny=TinyUtilityGate(**payload['spec'])
    tiny.load_state_dict(payload['state_dict'],strict=True)
    wrapper=VisualAnchoredAudioGate(tiny,payload['layout']).to(device).eval().requires_grad_(False)
    return wrapper,payload


class SelectedAudioQueryRouter(nn.Module):
    """Exact checkpoint endpoints. No residual/query-bank interpolation."""
    def __init__(self,audio_router,vision_router):
        super().__init__()
        self.audio_router,self.vision_router=audio_router,vision_router
        if audio_router.num_queries!=64 or vision_router.num_queries!=64:
            raise ValueError('Gate2 requires the fixed64-slot native interface')
        if audio_router.stage!=3 or vision_router.stage!=1:
            raise ValueError('explicit original stage3/stage1 checkpoints required')
        for module in (audio_router,vision_router):
            if module.training or any(p.requires_grad for p in module.parameters()):
                raise ValueError('original routing branches must be frozen and eval')

    @torch.inference_mode()
    def forward(self,visual,audio,timestamps,selection):
        if visual.ndim!=4 or timestamps.shape!=visual.shape[:2] or selection.shape!=visual.shape[:2] or selection.dtype!=torch.bool:
            raise ValueError('expected batch/time/native tokens and boolean per-window selection')
        if audio is not None and not isinstance(audio,AuxiliaryStream):
            raise ValueError('routing requires timestamped AuxiliaryStream, not unqualified audio tensors')
        allowed=causal_audio_mask(audio,timestamps) if audio is not None else None
        all_z,all_p=[],[]
        for b in range(len(visual)):
            selected=selection[b].clone()
            if audio is None:
                selected.zero_()
            else:
                selected &= allowed[b].any(-1)
            if not selected.any():
                z,p=self.vision_router(visual[b],None,timestamps[b])
            else:
                uniform=torch.equal(allowed[b],allowed[b,:1].expand_as(allowed[b]))
                if uniform:
                    # A common future/padding suffix is REMOVED before the
                    # original cross-attention, never left as zero-valued keys.
                    causal_audio=audio.features[b,:,allowed[b,0],:]
                    az,ap=self.audio_router(visual[b],causal_audio,timestamps[b])
                    if selected.all():
                        z,p=az,ap
                    else:
                        vz,vp=self.vision_router(visual[b],None,timestamps[b])
                        z=torch.where(selected[:,None,None],az,vz)
                        p=torch.where(selected[:,None,None],ap,vp)
                else:
                    # Variable token masks: retain each current causal state
                    # sequence and the actual previous decision timestamp.
                    z,p=self.vision_router(visual[b],None,timestamps[b])
                    z,p=z.clone(),p.clone()
                    for index in torch.where(selected)[0].tolist():
                        v=visual[b,index:index+1]
                        a=audio.features[b,index,allowed[b,index]][None]
                        if index==0:
                            az,ap=self.audio_router(v,a,timestamps[b,index:index+1])
                        else:
                            az,ap=self.audio_router(v.expand(2,-1,-1),a.expand(2,-1,-1),timestamps[b,index-1:index+1])
                        z[index],p[index]=az[-1],ap[-1]
            all_z.append(z);all_p.append(p)
        return torch.stack(all_z),torch.stack(all_p)
