"""Default-off exact branch wrapper and append-only causal online state."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from utility_gate.common import REPO, seed_all, write_json
from utility_gate.gate2_interfaces import AuxiliaryStream, ModalityGateOutput, SelectedAudioQueryRouter
from utility_gate.gate2_runtime import verify_protocol
from utility_gate.gate2_temporal import Gate2OnlineState
from utility_gate.integration import verify_disabled
from utility_gate.models import UtilityPrediction
from utility_gate.strict_runtime import StrictStreamingRuntime
from .calibration import SelectionRule, apply_selection
from .data_audit import GATE2, ROOT, read
from .interfaces import DualHeadInput, causal_features
from .model import DualHeadGate, load_dual_head


def slice_stream(stream,stop):
    if stream is None:return None
    return AuxiliaryStream(stream.features[:,:stop],stream.timestamps[:,:stop],stream.available[:,:stop],
        stream.modality_name,stream.token_mask[:,:stop] if stream.token_mask is not None else None)


class DualHeadRoutingWrapper(nn.Module):
    def __init__(self,audio_router,vision_router,model=None,rule=None,enabled=False):
        super().__init__()
        self.audio_router,self.vision_router,self.model=audio_router,vision_router,model
        self.rule=rule or SelectionRule();self.enabled=bool(enabled)
        self.selected=SelectedAudioQueryRouter(audio_router,vision_router)
        self.last_outputs=[]

    @torch.inference_mode()
    def forward(self,visual,audio,timestamps,decision_times=None,frame_pts=None):
        self.last_outputs=[]
        if not self.enabled:
            # Preserve the original unqualified legacy call exactly; no new
            # quality policy or gate is silently installed in the old entry.
            if visual.ndim==3:return self.audio_router(visual,audio,timestamps)
            zs=[];ps=[]
            for b in range(len(visual)):
                a=audio.features[b] if isinstance(audio,AuxiliaryStream) else audio[b] if audio is not None else None
                z,p=self.audio_router(visual[b],a,timestamps[b]);zs.append(z);ps.append(p)
            return torch.stack(zs),torch.stack(ps)
        if self.model is None or self.model.training:raise ValueError('explicit eval-mode dual-head model required')
        if audio is not None and not isinstance(audio,AuxiliaryStream):raise TypeError('enabled routing requires timestamped audio')
        single=visual.ndim==3
        if single:visual=visual[None];timestamps=timestamps[None]
        if frame_pts is not None and frame_pts.ndim==1:frame_pts=frame_pts[None]
        if decision_times is not None:
            counts=(timestamps<=decision_times[:,None]).sum(1)
            if not (counts==counts[0]).all() or int(counts[0])==0:raise ValueError('ragged/empty batch frontiers require per-stream calls')
            n=int(counts[0]);visual=visual[:,:n];timestamps=timestamps[:,:n];audio=slice_stream(audio,n)
            if frame_pts is not None:frame_pts=frame_pts[:,:n]
        decisions=[]
        for index in range(visual.shape[1]):
            inputs=DualHeadInput(visual[:,:index+1],timestamps[:,:index+1],timestamps[:,index],
                slice_stream(audio,index+1),frame_pts[:,:index+1] if frame_pts is not None else None)
            rule=replace(self.rule,control_step=index) if self.rule.mode=='random_rate_matched' else self.rule
            output=self.model(inputs,rule);self.last_outputs.append(output);decisions.append(output.selection_mask[:,0])
        choice=torch.stack(decisions,1)
        z,p=self.selected(visual,audio,timestamps,choice)
        return (z[0],p[0]) if single else (z,p)


class _DecisionAdapter:
    def __init__(self,model,rule):
        self.model,self.rule=model,rule;self.last_output=None;self.last_dual=None;self.control_step=0

    def predict(self,features):
        raw=self.model.forward_features(features)
        rule=replace(self.rule,control_step=self.control_step) if self.rule.mode=='random_rate_matched' else self.rule
        dual=apply_selection(raw,rule,self.model.head_status,self.model.decision_head)
        self.control_step+=1
        self.last_dual=dual
        encoded=dual.selection_mask.float()*2-1
        self.last_output=ModalityGateOutput(dual.selection_mask[:,None].float(),encoded[:,None],dual.availability_mask[:,None],('audio',))
        return UtilityPrediction(encoded[:,0],dual.correspondence_logit[:,0],dual.availability_mask[:,0])


class DualHeadOnlineState(Gate2OnlineState):
    def __init__(self,runtime,model,rule):
        self.adapter=_DecisionAdapter(model,rule)
        super().__init__(runtime,self.adapter,0.)

    @torch.inference_mode()
    def step(self,*args,**kwargs):
        z,p=super().step(*args,**kwargs)
        dual=self.adapter.last_dual;trace=self.trace[-1]
        trace['internal_selection_encoding']=trace.pop('utility_score')
        trace['internal_selection_threshold']=trace.pop('threshold')
        trace.update(correspondence_logit=float(dual.correspondence_logit[0,0]),utility_delta=float(dual.utility_delta[0,0]),
            selection_rule=self.adapter.rule.mode,head_status=dict(self.adapter.model.head_status),
            correspondence_calibrated_probability=False)
        return z,p


def smoke(root=ROOT):
    root=Path(root);seed_all(1234);runtime=StrictStreamingRuntime()
    baseline=verify_disabled(runtime);protocol=verify_protocol(runtime)
    fixture=torch.load(REPO/'artifacts/relevance_gate/m0-gate-disabled-final-20261022/fixture.pt',map_location='cpu',weights_only=True)
    visual=torch.cat([c['visual'] for c in fixture['chunks']]).to(runtime.device)
    audio=fixture['audio_info']['embeddings'].to(runtime.device);times=fixture['timestamps'].to(runtime.device)
    reference=read(GATE2/'temporal_model/selection.json')['primary']['checkpoint']
    model=DualHeadGate.from_gate2_utility(reference).to(runtime.device).eval()
    disabled=DualHeadRoutingWrapper(runtime.audio_router,runtime.vision_router)
    dz,dp=disabled(visual,audio,times)
    raw_z,raw_p=runtime.audio_router(visual,audio,times)
    assert torch.equal(dz,raw_z) and torch.equal(dp,raw_p)
    stream=AuxiliaryStream(audio[None],times[None,:,None].expand(1,len(times),len(audio[0])),
        torch.ones(1,len(times),dtype=torch.bool,device=runtime.device),'audio')
    from utility_gate.gate2_interfaces import load_gate2
    frozen_reference,_=load_gate2(reference,runtime.device)
    feature=causal_features(DualHeadInput(visual[None],times[None],times[-1:],stream))
    assert torch.equal(model.forward_features(feature).utility_delta[:,0],frozen_reference.predict(feature).utility_score)
    wrapper=DualHeadRoutingWrapper(runtime.audio_router,runtime.vision_router,model,SelectionRule('always_on'),True)
    z,p=wrapper(visual,stream,times)
    assert torch.equal(z,raw_z) and torch.equal(p,raw_p)
    accepted=runtime.generate(fixture['prompt_ids'],z)['generated_ids']
    assert accepted==fixture['generated_ids'].reshape(-1).tolist()
    wrapper.rule=SelectionRule('always_off');vz,vp=wrapper(visual,stream,times)
    expected_vz,expected_vp=runtime.vision_router(visual,None,times)
    assert torch.equal(vz,expected_vz) and torch.equal(vp,expected_vp)
    rejected=runtime.generate(fixture['prompt_ids'],vz)['generated_ids']
    wrapper.rule=SelectionRule('always_on');missing,_=wrapper(visual,None,times)
    assert torch.equal(missing,expected_vz)
    future=AuxiliaryStream(torch.cat((stream.features,torch.full((1,len(times),2,768),float('nan'),device=runtime.device)),2),
        torch.cat((stream.timestamps,times[None,:,None].expand(1,len(times),2)+100),2),stream.available,'audio')
    fz,fp=wrapper(visual,future,times)
    assert torch.equal(fz,z) and torch.equal(fp,p)
    trained_path=root/'model/utility_only/gate.pt'
    trained_check={'status':'not_run','reason':'utility-only diagnostic checkpoint not present'}
    if trained_path.exists():
        trained,payload=load_dual_head(trained_path,runtime.device)
        wrapper.model=trained;wrapper.rule=SelectionRule('always_off')
        tz,tp=wrapper(visual,stream,times)
        before=wrapper.last_outputs[-1]
        extended=AuxiliaryStream(torch.cat((stream.features,torch.full_like(stream.features[:,:1],float('nan'))),1),
            torch.cat((stream.timestamps,torch.full_like(stream.timestamps[:,:1],float(times[-1])+100)),1),
            torch.cat((stream.available,torch.ones_like(stream.available[:,:1])),1),'audio')
        ez,ep=wrapper(torch.cat((visual,torch.full_like(visual[:1],float('nan')))),extended,
            torch.cat((times,times[-1:]+100)),decision_times=times[-1:])
        after=wrapper.last_outputs[-1]
        assert torch.equal(tz,ez) and torch.equal(tp,ep) and torch.equal(tz,expected_vz)
        assert torch.equal(before.correspondence_logit,after.correspondence_logit)
        assert torch.equal(before.utility_delta,after.utility_delta)
        assert not any(p.requires_grad for p in trained.parameters())
        trained_check={'status':'passed_engineering_only','head_status':payload['head_status'],
            'future_visual_and_audio_invariant_r_u_selection_P_Z':True,'all_gate_parameters_frozen':True,
            'actual_policy':'always_off control; no fitted deployment threshold'}
    write_json(root/'audit/M2_real_GPU_smoke.json',{'legacy':baseline,'old_protocol':protocol,
        'default_off_exact_P_Z':True,'always_on_exact_P_Z':True,'all_off_missing_exact_vision':True,
        'future_audio_removed_before_both_heads_and_query':True,'accepted_generated_ids':accepted,
        'rejected_generated_ids':rejected,'native_slots':64,'new_correspondence_head_trained':False,
        'warmstarted_utility_exact_frozen_reference':True,
        'fresh_utility_only_checkpoint_native_compatibility':trained_check,
        'scope':'real captured-feature routing/generation engineering check, not new dual-head scientific evaluation'})
    print({'accepted':accepted,'rejected':rejected,'new_trained_dual_head':False})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT)
    smoke(p.parse_args().output)


if __name__=='__main__':main()
