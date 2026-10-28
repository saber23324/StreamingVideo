"""Synthetic engineering checks, never claimed as real modality performance."""
import copy
import dataclasses
import unittest

import torch
from torch import nn

from utility_gate.features import GateFeatures
from utility_gate.gate2_interfaces import AuxiliaryStream
from utility_gate.dual_head.calibration import SelectionRule, apply_selection, validate_calibration_rows
from utility_gate.dual_head.datasets import assert_source_isolation, training_support, validate_rows
from utility_gate.dual_head.diagnostics import paired_source_interval
from utility_gate.dual_head.integration import DualHeadOnlineState, DualHeadRoutingWrapper
from utility_gate.dual_head.interfaces import DualHeadInput, DualHeadOutput, causal_features
from utility_gate.dual_head.losses import DualSupervision, dual_loss
from utility_gate.dual_head.model import DualHeadGate


def fixture():
    torch.manual_seed(9)
    v=torch.randn(2,2,196,7);t=torch.tensor([[2.,4.],[2.,4.]])
    a=torch.randn(2,2,3,5)
    stream=AuxiliaryStream(a,t[:,:,None].expand(2,2,3),torch.ones(2,2,dtype=torch.bool),'audio')
    return DualHeadInput(v,t,t[:,-1],stream)


class DualHeadModelTests(unittest.TestCase):
    def test_distinct_signed_heads_and_no_unverified_default_selection(self):
        model=DualHeadGate(7,5,8).eval();model.utility_head.bias.data.fill_(-10)
        output=model(fixture())
        self.assertEqual(output.correspondence_logit.shape,(2,1))
        self.assertTrue((output.utility_delta<0).all())
        self.assertFalse(output.selection_mask.any())
        self.assertIsNot(model.correspondence_head,model.utility_head)
        self.assertFalse(output.diagnostics['correspondence_calibrated_probability'])

    def test_typed_inputs_reject_question_gold_and_source(self):
        model=DualHeadGate(7,5,8).eval()
        for key in ['question','answer','gold_nll','source_id']:
            with self.assertRaises(TypeError):model(fixture(),**{key:'forbidden'})
            with self.assertRaises(TypeError):DualHeadInput(**{**vars(fixture()),key:'forbidden'})
        with self.assertRaises(TypeError):model({'visual_tokens':fixture().visual_tokens})

    def test_future_audio_states_do_not_change_either_head_or_selection(self):
        model=DualHeadGate(7,5,8).eval();x=fixture();s=x.audio
        extra=torch.full((2,2,2,5),float('nan'))
        future=AuxiliaryStream(torch.cat((s.features,extra),2),torch.cat((s.timestamps,x.observation_times[:,:,None].expand(2,2,2)+10),2),s.available,'audio')
        old=model(x,SelectionRule('always_on'));new=model(dataclasses.replace(x,audio=future),SelectionRule('always_on'))
        for key in ('correspondence_logit','utility_delta','selection_mask'):
            torch.testing.assert_close(getattr(old,key),getattr(new,key),atol=0,rtol=0)

    def test_future_visual_and_audio_observations_removed_before_pooling(self):
        model=DualHeadGate(7,5,8).eval();x=fixture();s=x.audio
        future=AuxiliaryStream(torch.cat((s.features,torch.full((2,1,3,5),float('nan'))),1),
            torch.cat((s.timestamps,torch.full((2,1,3),99.)),1),torch.cat((s.available,torch.ones(2,1,dtype=torch.bool)),1),'audio')
        expanded=DualHeadInput(torch.cat((x.visual_tokens,torch.full((2,1,196,7),float('nan'))),1),
            torch.cat((x.observation_times,torch.full((2,1),99.)),1),x.decision_times,future)
        old=model(x);new=model(expanded)
        torch.testing.assert_close(old.correspondence_logit,new.correspondence_logit,atol=0,rtol=0)
        torch.testing.assert_close(old.utility_delta,new.utility_delta,atol=0,rtol=0)

    def test_missing_stale_and_nonfinite_audio_are_engineering_masks(self):
        model=DualHeadGate(7,5,8).eval();x=fixture()
        outputs=[model(dataclasses.replace(x,audio=None),SelectionRule('always_on'))]
        for times,features in [(x.audio.timestamps-10,x.audio.features),(x.audio.timestamps,torch.full_like(x.audio.features,float('nan')))]:
            bad=AuxiliaryStream(features,times,x.audio.available,'audio')
            outputs.append(model(dataclasses.replace(x,audio=bad),SelectionRule('always_on')))
        for o in outputs:
            self.assertFalse(o.availability_mask.any());self.assertFalse(o.selection_mask.any())
            self.assertTrue(torch.isfinite(o.utility_delta).all());self.assertTrue(torch.isfinite(o.correspondence_logit).all())

    def test_frozen_feature_gradients_and_independent_loss_heads(self):
        x=fixture();v=x.visual_tokens.requires_grad_();a=x.audio.features.requires_grad_()
        model=DualHeadGate(7,5,8);output=model(x)
        target=DualSupervision(torch.tensor([[1.],[0.]]),torch.ones(2,1,dtype=torch.bool),torch.ones(2,1),
            torch.tensor([[.2],[-.4]]),torch.ones(2,1,dtype=torch.bool),torch.ones(2,1))
        loss,info=dual_loss(output,target);loss.backward()
        self.assertIsNone(v.grad);self.assertIsNone(a.grad)
        self.assertGreater(float(model.correspondence_head.weight.grad.abs().sum()),0)
        self.assertGreater(float(model.utility_head.weight.grad.abs().sum()),0)
        self.assertEqual(info['corr_active'],2);self.assertEqual(info['utility_active'],2)

    def test_unknown_nan_targets_never_become_zero_supervision(self):
        model=DualHeadGate(7,5,8);output=model(fixture())
        labels=DualSupervision(torch.full((2,1),float('nan')),torch.zeros(2,1,dtype=torch.bool),torch.ones(2,1),
            torch.zeros(2,1),torch.ones(2,1,dtype=torch.bool),torch.ones(2,1))
        loss,info=dual_loss(output,labels);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertEqual(info['corr_active'],0)
        self.assertTrue(torch.equal(model.correspondence_head.weight.grad,torch.zeros_like(model.correspondence_head.weight)))

    def test_independent_multiselect_and_all_off_are_legal(self):
        out=DualHeadOutput(torch.tensor([[2.,2.,-2.]]),torch.tensor([[1.,1.,1.]]),torch.zeros(1,3,dtype=torch.bool),
            torch.ones(1,3,dtype=torch.bool),('audio','synthetic_imu_slot','synthetic_gui_slot'))
        status={'correspondence':'trained_semantic','utility':'trained_regression'}
        rule=SelectionRule('select_dual_threshold',corr_threshold=0.,utility_threshold=0.,fit_role='val')
        self.assertEqual(apply_selection(out,rule,status).selection_mask.tolist(),[[True,True,False]])
        self.assertFalse(apply_selection(out,SelectionRule(),status).selection_mask.any())

    def test_untrained_heads_and_unfitted_learned_policy_refuse_deployment(self):
        model=DualHeadGate(7,5,8).eval();x=fixture()
        for rule in [SelectionRule('select_corr',0.,fit_role='val'),SelectionRule('select_utility',utility_threshold=0.,fit_role='val'),
                     SelectionRule('select_dual_learned',decision_threshold=0.,fit_role='val')]:
            with self.assertRaises(ValueError):model(x,rule)

    def test_rate_matched_random_is_a_seeded_causal_control(self):
        model=DualHeadGate(7,5,8).eval();raw=model(fixture())
        rule=SelectionRule('random_rate_matched',fit_role='val',activation_rate=.5)
        a=apply_selection(raw,rule,model.head_status);b=apply_selection(raw,rule,model.head_status)
        self.assertTrue(torch.equal(a.selection_mask,b.selection_mask))
        off=apply_selection(raw,dataclasses.replace(rule,activation_rate=0.),model.head_status)
        on=apply_selection(raw,dataclasses.replace(rule,activation_rate=1.),model.head_status)
        self.assertFalse(off.selection_mask.any());self.assertTrue(torch.equal(on.selection_mask,raw.availability_mask))
        with self.assertRaises(ValueError):apply_selection(raw,dataclasses.replace(rule,fit_role='test'),model.head_status)

    def test_threshold_ties_reject_and_nan_threshold_raises(self):
        out=DualHeadOutput(torch.zeros(1,1),torch.zeros(1,1),torch.zeros(1,1,dtype=torch.bool),torch.ones(1,1,dtype=torch.bool))
        status={'utility':'trained_regression'}
        self.assertFalse(apply_selection(out,SelectionRule('select_utility',utility_threshold=0.,fit_role='val'),status).selection_mask.any())
        with self.assertRaises(ValueError):apply_selection(out,SelectionRule('select_utility',utility_threshold=float('nan'),fit_role='val'),status)


class DualHeadRoleTests(unittest.TestCase):
    def row(self):return {'dataset_role':'train','official_dataset':'AVQA','official_split':'train',
        'source_video_id':'youtube:train_video','target_family':'utility','supervision_scope':'measured_window',
        'eligible_audio':True,'nll_utility':.2}

    def test_all_evaluation_roles_and_prefix_targets_rejected_for_window_fitting(self):
        base=self.row();validate_rows([base],'train',set())
        for override in [{'dataset_role':'test'},{'official_split':'evaluation'},{'official_dataset':'LLP'},
                         {'official_dataset':'VGGSS'},{'supervision_scope':'prefix_feasibility'}]:
            with self.assertRaises(ValueError):validate_rows([{**base,**override}],'train',set())
        with self.assertRaises(ValueError):validate_rows([base],'train',{'train_video'})

    def test_semantic_training_requires_two_real_classes(self):
        row={**self.row(),'target_family':'correspondence','corr_target':1.}
        result=training_support([row],'dual_head')
        self.assertEqual(result['status'],'insufficient_data');self.assertEqual(result['semantic_negative_sources'],0)

    def test_unknown_cross_video_pair_cannot_be_rebranded_trusted_negative(self):
        row={**self.row(),'official_dataset':'AVSBench_S4','target_family':'correspondence','corr_target':0.,'corr_strength':'weak_pairing'}
        with self.assertRaises(ValueError):validate_rows([row],'train',set())

    def test_source_donor_and_byte_aliases_cannot_cross_roles(self):
        train=self.row();val={**train,'dataset_role':'val'}
        with self.assertRaises(ValueError):assert_source_isolation([train],[val])
        with self.assertRaises(ValueError):assert_source_isolation([{**train,'audio_source_video_id':'donor'}],[{**val,'source_video_id':'other','audio_source_video_id':'donor'}])
        with self.assertRaises(ValueError):assert_source_isolation([{**train,'media_sha256':'bytes'}],[{**val,'source_video_id':'other','media_sha256':'bytes'}])

    def test_calibration_underpowered_and_evaluation_roles_are_explicit(self):
        row={**self.row(),'dataset_role':'val','accuracy_utility':1}
        result=validate_calibration_rows([row]);self.assertEqual(result['status'],'insufficient_data')
        with self.assertRaises(ValueError):validate_calibration_rows([{**row,'official_split':'evaluation'}])

    def test_thousand_windows_are_not_thousand_independent_sources(self):
        result=paired_source_interval([0.]*1000,['one_video']*1000)
        self.assertEqual(result['status'],'insufficient_data');self.assertIsNone(result['ci95'])


class ToyRouter(nn.Module):
    def __init__(self,stage):
        super().__init__();self.stage=stage;self.num_queries=64
        self.weight=nn.Parameter(torch.ones(()),requires_grad=False);self.eval()

    def forward(self,v,a,t):
        score=v.mean(-1)
        if self.stage==3 and a is not None:score=score+a.mean((1,2))[:,None]*torch.linspace(-1,1,v.shape[1])
        p=score.softmax(-1)[:,None].expand(-1,64,-1)
        return p@v,p


class DualHeadIntegrationTests(unittest.TestCase):
    def test_disabled_accepted_rejected_missing_and_native_values(self):
        x=fixture();v=x.visual_tokens[:1];t=x.observation_times[:1]
        s=AuxiliaryStream(x.audio.features[:1],x.audio.timestamps[:1],x.audio.available[:1],'audio')
        audio,vision=ToyRouter(3),ToyRouter(1);model=DualHeadGate(7,5,8).eval()
        wrapper=DualHeadRoutingWrapper(audio,vision,model)
        z,p=wrapper(v,x.audio.features[:1],t);expected=audio(v[0],s.features[0],t[0])
        torch.testing.assert_close(z[0],expected[0],atol=0,rtol=0)
        wrapper.enabled=True;wrapper.rule=SelectionRule('always_on')
        z,p=wrapper(v,s,t);torch.testing.assert_close(z[0],expected[0],atol=0,rtol=0)
        torch.testing.assert_close(z,p@v,atol=0,rtol=0)
        wrapper.rule=SelectionRule();z,_=wrapper(v,s,t)
        torch.testing.assert_close(z[0],vision(v[0],None,t[0])[0],atol=0,rtol=0)
        wrapper.rule=SelectionRule('always_on');missing,_=wrapper(v,None,t)
        torch.testing.assert_close(missing,z,atol=0,rtol=0)

    def test_future_video_frontier_and_future_audio_leave_P_Z_unchanged(self):
        x=fixture();v=x.visual_tokens[:1];t=x.observation_times[:1]
        s=AuxiliaryStream(x.audio.features[:1],x.audio.timestamps[:1],x.audio.available[:1],'audio')
        wrapper=DualHeadRoutingWrapper(ToyRouter(3),ToyRouter(1),DualHeadGate(7,5,8).eval(),SelectionRule('always_on'),True)
        old_z,old_p=wrapper(v,s,t)
        future_s=AuxiliaryStream(torch.cat((s.features,torch.full((1,1,3,5),float('nan'))),1),
            torch.cat((s.timestamps,torch.full((1,1,3),99.)),1),torch.cat((s.available,torch.ones(1,1,dtype=torch.bool)),1),'audio')
        new_z,new_p=wrapper(torch.cat((v,torch.full((1,1,196,7),float('nan'))),1),future_s,
            torch.cat((t,torch.tensor([[99.]])),1),decision_times=torch.tensor([4.]))
        torch.testing.assert_close(new_z,old_z,atol=0,rtol=0);torch.testing.assert_close(new_p,old_p,atol=0,rtol=0)

    def test_online_history_is_immutable_and_reports_actual_two_head_scores(self):
        runtime=type('R',(),{'audio_router':ToyRouter(3),'vision_router':ToyRouter(1)})()
        model=DualHeadGate(7,768,8).eval();state=DualHeadOnlineState(runtime,model,SelectionRule('always_on'))
        v=torch.randn(196,7);a=torch.randn(4,768)
        state.step(v,a,2.,True);previous=state.bank().clone()
        state.step(v,a,4.,True)
        torch.testing.assert_close(previous[0],state.bank()[0],atol=0,rtol=0)
        self.assertIn('utility_delta',state.trace[-1]);self.assertIn('correspondence_logit',state.trace[-1])
        self.assertNotIn('utility_score',state.trace[-1])


if __name__=='__main__':unittest.main()
