"""Behavioral checks on synthetic inputs; these are not research results."""
import dataclasses
import math
from pathlib import Path
import tempfile
import unittest

import torch

from utility_gate.common import validate_splits
from utility_gate.features import GateFeatures, feature_payload, load_feature_payload, prefix_features
from utility_gate.labels import build_labels, complementarity, verify_pair
from utility_gate.runtime import parse_prediction, score_answer
from utility_gate.models import TinyUtilityGate
from utility_gate.metrics import fit_threshold, random_selector, regression_metrics, selection_metrics
from utility_gate.train import objective
from utility_gate.policy import UtilityRoutingPolicy, CausalStreamRouter
from utility_gate.models import UtilityPrediction
from utility_gate.streaming import replace_one_window


def synthetic_row(index, split='train', vision=False, audio=True, utility=1.):
    v = 2. + max(utility, 0.)
    a = v-utility
    branch = lambda c, n: {'correct': c, 'gold_nll': n, 'nll_protocol': 'synthetic same protocol',
        'visual_tokens': 128, 'total_input_tokens': 140, 'parser_valid': True}
    return {'question_id': f'{split}-q{index}', 'source_video_id': f'{split}-v{index}', 'videoID': f'v{index}',
        'video_path': f'/synthetic/{split}-{index}.mp4', 'media_sha256': f'{split}-bytes{index}',
        'dataset': 'SYNTHETIC', 'split': split, 'task_type': 'visual', 'answer': 'A', 'query_timestamp_s': 2.,
        'feature_cache_key': 'SYNTHETIC', 'feature_cache_path': '/synthetic/tensor.pt',
        'config_fingerprint': 'SYNTHETIC', 'question_modality': 'visual', 'token_budget_per_frame': 64,
        'timestamps_s': [0.,2.], 'frame_pts_s': [0.,2.], 'audio_missing_fraction': .5, 'mode': 'prefix',
        'branches': {'audio': branch(audio,a), 'vision': branch(vision,v), 'full_vision': branch(True,1.)},
        'accuracy_utility': int(audio)-int(vision), 'nll_utility': utility}


class UtilityFeatureTests(unittest.TestCase):
    def test_prefix_features_cannot_depend_on_future_frames_audio_or_questions(self):
        v, a = torch.randn(4, 3, 7), torch.randn(4, 5, 4)
        expected = prefix_features(v, a, [False,True,True,True], [0.,2.,4.,6.])
        v[2:] = 999.; a[2:] = -999.
        changed = prefix_features(v, a, [False,True,False,False], [0.,2.,400.,600.])
        for name in ('visual', 'audio', 'context', 'valid_audio'):
            self.assertTrue(torch.equal(getattr(expected,name)[:2],getattr(changed,name)[:2]))

    def test_missing_and_nonfinite_audio_are_masked_without_nan(self):
        v, a = torch.ones(3, 2, 4), torch.ones(3, 2, 5)
        a[0] = float('nan'); a[2] = float('inf')
        value = prefix_features(v,a,[False,True,True],[0.,1.,2.])
        self.assertEqual(value.valid_audio.tolist(), [False,True,True])
        self.assertTrue(torch.isfinite(value.audio).all())
        self.assertTrue(torch.equal(value.audio[0],torch.zeros(5)))

    def test_deployment_schema_rejects_qa_and_oracle_fields(self):
        names = {f.name for f in dataclasses.fields(GateFeatures)}
        self.assertEqual(names, {'visual','audio','context','valid_audio'})
        value = prefix_features(torch.ones(1,2,4),torch.ones(1,2,5),[True],[1.])
        payload=feature_payload(value)
        for forbidden in ('gold_answer','question','gold_nll','correct','utility_label','source_video_id'):
            with self.assertRaisesRegex(ValueError, 'unexpected feature field'):
                load_feature_payload({**payload,forbidden:torch.ones(1)})

    def test_invalid_order_and_batch_shapes_fail(self):
        with self.assertRaisesRegex(ValueError,'sorted'):
            prefix_features(torch.ones(2,2,4),torch.ones(2,2,5),[True,True],[2.,1.])
        with self.assertRaisesRegex(ValueError,'batch size'):
            GateFeatures(torch.ones(2,4),torch.ones(1,5),torch.ones(2,4),torch.ones(2,dtype=torch.bool)).validate()


class UtilitySplitTests(unittest.TestCase):
    def test_source_and_alias_and_byte_duplicates_cannot_cross_splits(self):
        original=synthetic_row(1)
        for key in ('source_video_id','video_path','media_sha256'):
            other=synthetic_row(2,'test'); other[key]=original[key]
            with self.assertRaisesRegex(ValueError,'crosses split'):
                validate_splits([original,other])

    def test_multiple_questions_of_same_video_must_stay_together(self):
        first=synthetic_row(1); second={**first,'question_id':'another_question'}
        self.assertEqual(validate_splits([first,second])['source_videos'],1)
        with self.assertRaisesRegex(ValueError,'duplicate question'):
            validate_splits([first,first])


class UtilityScoringTests(unittest.TestCase):
    def generation(self):
        return {'text':'A','option_logits':dict(zip('ABCD',[2.,1.,0.,-1.])),
                'option_logprobs_full_vocab':dict(zip('ABCD',[-2.,-3.,-4.,-5.]))}

    def test_correctness_and_candidate_normalized_gold_nll(self):
        result=score_answer(self.generation(),'A')
        self.assertTrue(result['correct'])
        self.assertAlmostEqual(result['gold_nll'],math.log(sum(math.exp(x) for x in [2,1,0,-1]))-2)
        self.assertEqual(result['gold_nll_full_vocab'],2.)

    def test_invalid_parse_stays_wrong_not_forced_to_option_argmax(self):
        generation=self.generation(); generation['text']='The answer is maybe A or B'
        result=score_answer(generation,'A')
        self.assertFalse(result['correct']); self.assertFalse(result['parser_valid'])
        self.assertEqual(result['option_prediction'],'A')

    def test_nonfinite_logits_or_invalid_answers_fail(self):
        generation=self.generation(); generation['option_logits']['A']=float('nan')
        with self.assertRaisesRegex(ValueError,'nonfinite'): score_answer(generation,'A')
        with self.assertRaisesRegex(ValueError,'gold answer'): score_answer(self.generation(),'AB')

    def test_same_local_mlvu_parser_contract(self):
        self.assertEqual(parse_prediction('A) option','mlvu_dev'),'A')
        self.assertIsNone(parse_prediction('A) option','streamingbench'))


class UtilityLabelTests(unittest.TestCase):
    def test_quadrants_and_oracle_are_recomputed(self):
        rows=[synthetic_row(1,vision=True,audio=True),synthetic_row(2,vision=False,audio=True),
              synthetic_row(3,vision=True,audio=False,utility=-1),synthetic_row(4,vision=False,audio=False)]
        result=complementarity(rows)
        self.assertEqual(result['oracle_accuracy'],.75)
        for name in ('both_correct','audio_only_win','vision_only_win','both_wrong'):
            self.assertEqual(result[name],1)

    def test_label_ties_stay_ambiguous_and_test_cannot_set_delta(self):
        rows=[synthetic_row(1,vision=True,audio=True,utility=.1),synthetic_row(2,utility=1.)]
        labels,config=build_labels(rows+[synthetic_row(3,'test',utility=100.)])
        changed,again=build_labels(rows+[synthetic_row(3,'test',utility=-100.)])
        self.assertEqual(config,again)
        self.assertEqual(labels[0]['label_state'],'ambiguous')
        self.assertEqual(labels[1]['label_state'],'audio_better')
        self.assertEqual(config['fit_split'],'train')

    def test_pair_mismatches_and_future_frames_fail(self):
        row=synthetic_row(1); row['branches']['vision']['visual_tokens']=127
        with self.assertRaisesRegex(ValueError,'budgets'):verify_pair(row)
        row=synthetic_row(1); row['timestamps_s']=[0.,3.]
        with self.assertRaisesRegex(ValueError,'future'):verify_pair(row)

    def test_corrupted_stored_utility_is_not_accepted(self):
        row=synthetic_row(1);row['nll_utility']=999.
        with self.assertRaisesRegex(ValueError,'recomputed'):build_labels([row])


def metric_row(index, split='val', vision=False, audio=True, utility=1.):
    original=synthetic_row(index,split,vision,audio,utility)
    return {**original,'audio_correct':audio,'vision_correct':vision,
            'audio_gold_nll':original['branches']['audio']['gold_nll'],
            'vision_gold_nll':original['branches']['vision']['gold_nll']}


class UtilityTrainingTests(unittest.TestCase):
    def test_only_gate_receives_gradients_on_frozen_cached_features(self):
        visual=torch.randn(3,7,requires_grad=True)
        audio=torch.randn(3,5,requires_grad=True)
        context=torch.randn(3,4,requires_grad=True)
        features=GateFeatures(visual,audio,context,torch.tensor([True,False,True]))
        model=TinyUtilityGate(7,5,hidden=8)
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        loss=objective(model,features,torch.tensor([1.,0.,-1.]),torch.tensor([1.,0.,-1.]),torch.ones(3),.2)
        loss.backward();optimizer.step()
        self.assertIsNone(visual.grad);self.assertIsNone(audio.grad);self.assertIsNone(context.grad)
        self.assertTrue(any(not torch.equal(v,before[k]) for k,v in model.state_dict().items()))

    def test_batched_missing_audio_and_predict_schema(self):
        model=TinyUtilityGate(7,5,hidden=8).eval()
        x=GateFeatures(torch.randn(2,7),torch.zeros(2,5),torch.zeros(2,4),torch.zeros(2,dtype=torch.bool))
        pred=model.predict(x)
        self.assertEqual(pred.utility_score.shape,(2,))
        self.assertTrue(torch.isfinite(pred.utility_score).all())
        self.assertFalse(pred.valid_audio.any())
        with self.assertRaisesRegex(TypeError,'schema'):
            model({'gold_answer':'A'})
        model.train()
        with self.assertRaisesRegex(RuntimeError,'eval'):
            model.predict(x)


class UtilityThresholdTests(unittest.TestCase):
    def test_threshold_is_validation_only_and_separates_known_branch_risks(self):
        rows=[metric_row(1),metric_row(2,vision=True,audio=False,utility=-1.)]
        result=fit_threshold(rows,[1.,-1.],[True,True])
        choices=torch.tensor([1.,-1.]).numpy()>result['threshold']
        self.assertEqual(choices.tolist(),[True,False])
        metrics=selection_metrics(rows,choices)
        self.assertEqual(metrics['accuracy'],1.)
        self.assertEqual(metrics['false_allow_harmful'],0.)
        with self.assertRaisesRegex(ValueError,'validation'):
            fit_threshold([{**r,'split':'test'} for r in rows],[1.,-1.],[True,True])

    def test_ties_are_strict_and_missing_audio_stays_rejected(self):
        rows=[metric_row(1),metric_row(2,vision=True,audio=False,utility=-1.)]
        result=fit_threshold(rows,[0.,0.],[False,True])
        self.assertEqual(((torch.tensor([0.,0.]).numpy()>result['threshold']) & [False,True]).tolist(),[False,False])
        self.assertEqual(result['comparison'],'strict_greater_than')

    def test_random_control_matches_on_count_and_is_deterministic(self):
        rows=[metric_row(i) for i in range(8)]
        valid=[False]+[True]*7
        a=random_selector(rows,valid,3);b=random_selector(rows,valid,3)
        self.assertTrue((a==b).all());self.assertEqual(int(a.sum()),3);self.assertFalse(a[0])

    def test_raw_utility_is_not_claimed_to_be_a_calibrated_probability(self):
        rows=[metric_row(1),metric_row(2,vision=True,audio=False,utility=-1.)]
        result=regression_metrics(rows,[10.,-10.],[True,False])
        self.assertEqual(result['discordant_QA_classification']['auroc'],1.)
        self.assertIsNone(result['discordant_QA_classification']['brier'])


class ConstantUtilityGate(torch.nn.Module):
    def predict(self, features):
        return UtilityPrediction(torch.zeros(len(features.visual)),torch.zeros(len(features.visual)),features.valid_audio)


class UtilityRoutingTests(unittest.TestCase):
    def setUp(self):
        from streamingtom.audio_bottleneck import AudioConditionedBottleneck
        torch.manual_seed(1234)
        self.audio=AudioConditionedBottleneck(7,num_queries=64,hidden_size=16,num_heads=4,stage=3,
            latent_norm='none',value_mode='native',architecture='phase4',attention_temperature=.05).eval().requires_grad_(False)
        self.vision=AudioConditionedBottleneck(7,num_queries=64,hidden_size=16,num_heads=4,stage=1,
            latent_norm='none',value_mode='native',architecture='phase4',attention_temperature=.05).eval().requires_grad_(False)
        self.gate=ConstantUtilityGate().eval().requires_grad_(False)
        self.v,self.a=torch.randn(3,196,7),torch.randn(3,4,768)
        self.t=torch.tensor([0.,2.,4.])

    def features(self,valid=True):
        return GateFeatures(self.v.mean((0,1))[None],self.a.mean((0,1))[None],torch.zeros(1,4),torch.tensor([valid]))

    def test_disabled_preserves_exact_original_route_without_gate_inputs(self):
        original=self.audio(self.v,self.a,self.t)
        result=UtilityRoutingPolicy(self.audio)(self.v,self.a,self.t)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(original,result)))

    def test_accept_reject_and_missing_audio_are_whole_original_branches(self):
        for threshold,expected in [(-1,self.audio(self.v,self.a,self.t)),(0,self.vision(self.v,None,self.t))]:
            policy=UtilityRoutingPolicy(self.audio,self.vision,self.gate,threshold,True)
            result=policy(self.v,self.a,self.t,self.features())
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(expected,result)))
        policy=UtilityRoutingPolicy(self.audio,self.vision,self.gate,-1,True)
        result=policy(self.v,None,self.t,self.features(True))
        expected=self.vision(self.v,None,self.t)
        self.assertFalse(policy.last_decision.accepted.any())
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(expected,result)))

    def test_streaming_retains_original_delta_time_context_and_never_revises_history(self):
        policy=UtilityRoutingPolicy(self.audio,self.vision,self.gate,-1,True)
        stream=CausalStreamRouter(policy)
        stream.step(self.v[0],self.a[0],0.,True,0.)
        first=stream.token_bank().clone()
        stream.step(self.v[1],self.a[1],2.,True,2.)
        expected,_=self.audio(self.v[:2],self.a[:2],self.t[:2])
        self.assertTrue(torch.allclose(stream.token_bank(),expected,atol=1e-6,rtol=1e-5))
        stream.step(self.v[2]*100,self.a[2]*-100,4.,True,4.)
        self.assertTrue(torch.equal(stream.token_bank()[:1],first))
        self.assertTrue(all(not r['historical_tokens_revised'] for r in stream.trace))

    def test_streaming_invalid_audio_falls_back_and_future_frame_is_rejected(self):
        stream=CausalStreamRouter(UtilityRoutingPolicy(self.audio,self.vision,self.gate,-1,True))
        z,_=stream.step(self.v[0],None,0.,False,0.)
        expected,_=self.vision(self.v[:1],None,self.t[:1])
        self.assertTrue(torch.equal(z,expected[0]));self.assertFalse(stream.trace[0]['accepted'])
        with self.assertRaisesRegex(ValueError,'future frame'):
            stream.step(self.v[1],self.a[1],2.,True,3.)
        with self.assertRaisesRegex(ValueError,'increasing'):
            stream.step(self.v[0],self.a[0],0.,True,0.)


class TemporalCreditTests(unittest.TestCase):
    def test_replacement_changes_only_the_named_window_and_preserves_baseline(self):
        audio=torch.arange(4*64*7).reshape(4,64,7).float()
        vision=audio*-2.
        original=audio.clone()
        replaced=replace_one_window(audio,vision,2)
        self.assertTrue(torch.equal(audio,original))
        self.assertTrue(torch.equal(replaced[2],vision[2]))
        self.assertTrue(torch.equal(replaced[[0,1,3]],audio[[0,1,3]]))
        self.assertEqual(replaced.shape,audio.shape)

    def test_replacement_rejects_slot_budget_mismatches(self):
        with self.assertRaisesRegex(ValueError,'matching'):
            replace_one_window(torch.zeros(3,64,7),torch.zeros(3,63,7),1)


if __name__ == '__main__':
    unittest.main()
