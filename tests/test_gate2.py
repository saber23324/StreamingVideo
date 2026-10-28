"""Gate2 interface/causality checks on explicitly synthetic engineering inputs."""
import dataclasses
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from utility_gate.gate2_data import original_source,select_records
from utility_gate.gate2_interfaces import (AuxiliaryStream,ModalityGateOutput,SelectedAudioQueryRouter,
    VisualAnchoredAudioGate,causal_audio_pool,independent_gates,visual_anchored_features)
from utility_gate.models import TinyUtilityGate
from utility_gate.gate2_experiment import stable_threshold,train_gate2
from utility_gate.gate2_temporal import Gate2OnlineState


class IndependentGateTests(unittest.TestCase):
    def test_multiple_and_none_are_valid_without_a_modality_softmax(self):
        names=('audio','synthetic_b','synthetic_c')
        logits=torch.tensor([[[8.,8.,-8.]],[[8.,-8.,-8.]],[[8.,8.,8.]]])
        mask=torch.tensor([[[True,True,True]],[[True,True,True]],[[False,False,False]]])
        result=independent_gates(logits,mask,names)
        self.assertEqual(result.hard_selection(0).tolist(),[[[True,True,False]],[[True,False,False]],[[False,False,False]]])
        self.assertGreater(float(result.gates[0].sum()),1.)
        self.assertTrue(torch.equal(result.gates[2],torch.zeros_like(result.gates[2])))

    def test_nonfinite_logits_and_missing_modalities_have_finite_zero_outputs(self):
        logits=torch.tensor([[[float('nan'),float('inf'),2.]]],requires_grad=True)
        result=independent_gates(logits,torch.ones(1,1,3,dtype=torch.bool),('audio','synthetic_b','synthetic_c'))
        self.assertTrue(torch.isfinite(result.gates).all() and torch.isfinite(result.logits).all())
        self.assertEqual(result.availability.tolist(),[[[False,False,True]]])
        result.gates.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad[0,0,:2].tolist(),[0.,0.])

    def test_deployed_stream_schema_excludes_question_and_oracle_fields(self):
        names={f.name for f in dataclasses.fields(AuxiliaryStream)}
        self.assertFalse(names & {'gold_answer','question','correct','gold_nll','source_video_id','utility_label'})
        with self.assertRaises(TypeError):
            AuxiliaryStream(features=torch.zeros(1,1,1,5),timestamps=torch.zeros(1,1,1),
                            available=torch.ones(1,1,dtype=torch.bool),modality_name='audio',gold_answer='A')


class CausalAnchorTests(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(2)
        visual=torch.randn(2,3,196,7)
        audio=torch.randn(2,3,4,5)
        time=torch.tensor([[2.,4.,6.],[2.,4.,6.]])
        stream=AuxiliaryStream(audio,time[:,:,None].expand(2,3,4),torch.ones(2,3,dtype=torch.bool),'audio')
        return visual,time,stream

    def test_future_audio_is_removed_before_gate_features(self):
        visual,time,stream=self.fixture()
        expected=visual_anchored_features(visual,time,stream)
        future=AuxiliaryStream(torch.cat((stream.features,torch.full((2,3,2,5),float('nan'))),2),
            torch.cat((stream.timestamps,time[:,:,None].expand(2,3,2)+100.),2),stream.available,'audio')
        actual=visual_anchored_features(visual,time,future)
        for field in ('visual','audio','context','valid_audio'):
            self.assertTrue(torch.equal(getattr(expected,field),getattr(actual,field)))

    def test_future_visual_audio_and_questions_cannot_change_current_gate(self):
        visual,time,stream=self.fixture()
        gate=VisualAnchoredAudioGate(TinyUtilityGate(14,10,hidden=8)).eval()
        original=gate(visual,time,stream)
        altered=visual.clone();altered[:,2]=1000.
        audio=stream.features.clone();audio[:,2]=-1000.
        future=AuxiliaryStream(audio,stream.timestamps,stream.available,'audio')
        changed=gate(altered,time,future)
        self.assertTrue(torch.equal(original.gates[:,:2],changed.gates[:,:2]))
        # A question/answer is not accepted by either public gate API.
        with self.assertRaises(TypeError):gate(visual,time,stream,question='future question')

    def test_availability_and_token_masks_prevent_nonfinite_pooling(self):
        visual,time,stream=self.fixture()
        audio=stream.features.clone();audio[0]=float('nan')
        available=stream.available.clone();available[0]=False
        pooled,mask=causal_audio_pool(AuxiliaryStream(audio,stream.timestamps,available,'audio'),time)
        self.assertTrue(torch.isfinite(pooled).all());self.assertFalse(mask[0].any())
        self.assertTrue(torch.equal(pooled[0],torch.zeros_like(pooled[0])))

    def test_only_small_gate_gets_gradient(self):
        visual,time,stream=self.fixture()
        visual.requires_grad_(True);stream.features.requires_grad_(True)
        gate=VisualAnchoredAudioGate(TinyUtilityGate(14,10,hidden=8)).train()
        gate(visual,time,stream).gates.sum().backward()
        self.assertIsNone(visual.grad);self.assertIsNone(stream.features.grad)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in gate.parameters()))

    def test_pooled_ablation_uses_the_same_causal_input_contract(self):
        visual,time,stream=self.fixture()
        gate=VisualAnchoredAudioGate(TinyUtilityGate(7,5,hidden=8),'pooled_history').eval()
        output=gate(visual,time,stream)
        self.assertEqual(tuple(output.gates.shape),(2,3,1))
        self.assertTrue(torch.isfinite(output.gates).all())


class SelectedQueryTests(unittest.TestCase):
    def setUp(self):
        from streamingtom.audio_bottleneck import AudioConditionedBottleneck
        torch.manual_seed(4)
        args=dict(visual_dim=7,num_queries=64,hidden_size=16,num_heads=4,latent_norm='none',value_mode='native',architecture='phase4',attention_temperature=.05)
        self.audio=AudioConditionedBottleneck(stage=3,**args).eval().requires_grad_(False)
        self.vision=AudioConditionedBottleneck(stage=1,**args).eval().requires_grad_(False)
        self.router=SelectedAudioQueryRouter(self.audio,self.vision)
        self.v=torch.randn(2,3,196,7)
        self.a=torch.randn(2,3,4,768)
        self.t=torch.tensor([[2.,4.,6.],[2.,4.,6.]])
        self.stream=AuxiliaryStream(self.a,self.t[:,:,None].expand(2,3,4),torch.ones(2,3,dtype=torch.bool),'audio')

    def test_all_on_is_exact_original_audio_and_all_off_exact_vision(self):
        for enabled,module in [(True,self.audio),(False,self.vision)]:
            z,p=self.router(self.v,self.stream,self.t,torch.full((2,3),enabled,dtype=torch.bool))
            self.assertEqual(tuple(z.shape),(2,3,64,7));self.assertEqual(tuple(p.shape),(2,3,64,196))
            self.assertTrue(torch.allclose(p.sum(-1),torch.ones(2,3,64),atol=1e-6))
            for b in range(2):
                expected=module(self.v[b],self.a[b] if enabled else None,self.t[b])
                self.assertTrue(torch.equal(z[b],expected[0]));self.assertTrue(torch.equal(p[b],expected[1]))

    def test_mixed_selection_and_missing_audio_do_not_reuse_audio_state(self):
        selection=torch.tensor([[True,False,True],[False,True,False]])
        z,p=self.router(self.v,self.stream,self.t,selection)
        for b in range(2):
            av,ap=self.audio(self.v[b],self.a[b],self.t[b]);vv,vp=self.vision(self.v[b],None,self.t[b])
            self.assertTrue(torch.equal(z[b],torch.where(selection[b,:,None,None],av,vv)))
        fallback,_=self.router(self.v,None,self.t,torch.ones_like(selection))
        for b in range(2):self.assertTrue(torch.equal(fallback[b],self.vision(self.v[b],None,self.t[b])[0]))

    def test_future_encoded_states_cannot_change_routed_tokens(self):
        mask=torch.ones(2,3,dtype=torch.bool)
        expected=self.router(self.v,self.stream,self.t,mask)
        future=AuxiliaryStream(torch.cat((self.a,torch.randn(2,3,2,768)*1000),2),
            torch.cat((self.stream.timestamps,self.t[:,:,None].expand(2,3,2)+1.),2),mask,'audio')
        actual=self.router(self.v,future,self.t,mask)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(expected,actual)))

    def test_variable_token_masks_and_stale_inputs_are_safe(self):
        token_mask=torch.ones_like(self.stream.timestamps,dtype=torch.bool);token_mask[0,1,2:]=False
        stream=AuxiliaryStream(self.a,self.stream.timestamps,self.stream.available,'audio',token_mask)
        z,p=self.router(self.v,stream,self.t,torch.ones(2,3,dtype=torch.bool))
        self.assertTrue(torch.isfinite(z).all() and torch.isfinite(p).all())
        stale=AuxiliaryStream(self.a,self.stream.timestamps-3.,self.stream.available,'audio')
        fallback,_=self.router(self.v,stale,self.t,torch.ones(2,3,dtype=torch.bool))
        for b in range(2):self.assertTrue(torch.equal(fallback[b],self.vision(self.v[b],None,self.t[b])[0]))

    def test_empty_audio_state_sequence_has_a_finite_vision_fallback(self):
        empty=AuxiliaryStream(torch.zeros(2,3,0,768),torch.zeros(2,3,0),self.stream.available,'audio')
        z,p=self.router(self.v,empty,self.t,torch.ones(2,3,dtype=torch.bool))
        self.assertTrue(torch.isfinite(z).all() and torch.isfinite(p).all())
        for b in range(2):self.assertTrue(torch.equal(z[b],self.vision(self.v[b],None,self.t[b])[0]))

    def test_chronological_state_rejects_future_or_missing_audio_without_revising_history(self):
        runtime=SimpleNamespace(audio_router=self.audio,vision_router=self.vision)
        gate=VisualAnchoredAudioGate(TinyUtilityGate(14,1536,hidden=8)).eval().requires_grad_(False)
        state=Gate2OnlineState(runtime,gate,-1000.)
        state.step(self.v[0,0],self.a[0,0],2.,True,2.,2.)
        first=state.bank().clone()
        z,_=state.step(self.v[0,1],self.a[0,1],4.,True,5.,4.)
        self.assertFalse(state.trace[-1]['accepted'])
        expected=self.vision(self.v[0,1:2],None,torch.tensor([4.]))[0][0]
        self.assertTrue(torch.equal(z,expected));self.assertTrue(torch.equal(state.bank()[:1],first))
        z,_=state.step(self.v[0,2],None,6.,True,6.,6.)
        self.assertFalse(state.trace[-1]['availability'])
        self.assertTrue(torch.isfinite(z).all());self.assertTrue(torch.equal(state.bank()[:1],first))


class OfficialSourceSplitTests(unittest.TestCase):
    def records(self,ids):
        return [{'id':i,'video_name':f'src{i:08d}_000000','question_relation':'Both','answer':0} for i in ids]

    def test_source_overlap_is_removed_and_selection_does_not_use_answers(self):
        train=self.records(range(20));evaluation=self.records([0]+list(range(20,30)))
        available={r['video_name'] for r in train+evaluation}
        picked,audit=select_records(train,evaluation,available,{'src00000001'},{'train':6,'val':2,'test':3})
        groups={split:{original_source(x['record']['video_name'])[0] for x in picked if x['split']==split} for split in ('train','val','test')}
        self.assertFalse(groups['train']&groups['val'] or groups['train']&groups['test'] or groups['val']&groups['test'])
        self.assertNotIn('src00000000',groups['train']|groups['val'])
        self.assertNotIn('src00000001',set.union(*groups.values()))
        for r in train+evaluation:r['answer']=3
        again,_=select_records(train,evaluation,available,{'src00000001'},{'train':6,'val':2,'test':3})
        self.assertEqual([(x['split'],x['record']['id']) for x in picked],[(x['split'],x['record']['id']) for x in again])

    def test_evaluation_targets_cannot_enter_training_function(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            train=root/'train.jsonl';val=root/'val.jsonl'
            train.write_text(json.dumps({'split':'test','official_split':'evaluation'})+'\n')
            val.write_text(json.dumps({'split':'val','official_split':'train'})+'\n')
            with self.assertRaisesRegex(ValueError,'official-training-source'):
                train_gate2(train,val,root/'must_not_exist')
            self.assertFalse((root/'must_not_exist').exists())

    def test_unavailable_scores_cannot_shift_validation_threshold_and_test_is_rejected(self):
        rows=[{'split':'val','accuracy_utility':1,'nll_utility':1.,'audio_correct':True,'vision_correct':False,
            'audio_gold_nll':.1,'vision_gold_nll':1.1},
            {'split':'val','accuracy_utility':0,'nll_utility':0.,'audio_correct':True,'vision_correct':True,
             'audio_gold_nll':.1,'vision_gold_nll':.1}]
        first=stable_threshold(rows,[1.,100.],[True,False])
        again=stable_threshold(rows,[1.,-100.],[True,False])
        self.assertEqual(first['threshold'],again['threshold'])
        with self.assertRaisesRegex(ValueError,'validation'):
            stable_threshold([{**r,'split':'test'} for r in rows],[1.,0.],[True,False])


if __name__=='__main__':unittest.main()
