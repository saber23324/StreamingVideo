"""New authorization/labels/masks: official Validation is not official Test."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from utility_gate.features import GateFeatures
from utility_gate.grounding.llp_parser import labels_at
from utility_gate.dual_head.model import DualHeadGate
from utility_gate.dual_head.losses import dual_loss
from utility_gate.dual_head.proxy.data import dense_target
from utility_gate.dual_head.proxy.training import assert_isolated, guard_rows, make_labels
from utility_gate.dual_head.proxy.evaluation import apply_rule


class ProxyDenseLabels(unittest.TestCase):
    def test_same_class_same_bin_positive(self):
        label=labels_at([(0,1,'dog')],[(0,1,'dog')],1)
        self.assertEqual(dense_target(label)[0],1.)

    def test_different_bin_same_category_is_not_joint(self):
        label=labels_at([(0,1,'dog')],[(1,2,'dog')],2)
        self.assertFalse(label['current_bin_joint']);self.assertFalse(label['causal_window_joint'])
        self.assertEqual(dense_target(label)[0],0.)
        self.assertIn('proxy_negative',dense_target(label)[1])

    def test_previous_joint_and_neither_remain_unknown(self):
        previous=labels_at([(0,1,'dog')],[(0,2,'dog')],2)
        self.assertIsNone(dense_target(previous)[0])
        self.assertIsNone(dense_target(labels_at([],[],1))[0])

    def test_boundary_is_half_open(self):
        label=labels_at([(1,2,'dog')],[(0,1,'dog')],1)
        self.assertFalse(label['current_bin_joint'])
        self.assertEqual(label['audio_categories'],[])


class ProxyRolesAndMasks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        (self.root/'audit').mkdir()
        (self.root/'protocol.json').write_text(json.dumps({'authorization':'LLP official Validation explicitly authorized'}))
        (self.root/'audit/source_roles.json').write_text(json.dumps({'a':'train','b':'val'}))
        self.base={'row_id':'a1','kind':'LLP','split':'train','official_split':'val','source_video_id':'a',
            'corr_target':0.,'label_strength':'official_dense_nonjoint_proxy_negative','corr_weight':.75,
            'utility_target':None,'quality_valid':True}

    def tearDown(self):self.tmp.cleanup()

    def test_validation_repurposing_allowed_test_always_rejected(self):
        guard_rows([self.base],'train',self.root)
        with self.assertRaises(ValueError):guard_rows([{**self.base,'official_split':'test'}],'train',self.root)
        with self.assertRaises(ValueError):guard_rows([{**self.base,'split':'test'}],'test',self.root)

    def test_proxy_provenance_and_cross_role_donor_enforced(self):
        with self.assertRaises(ValueError):guard_rows([{**self.base,'label_strength':'semantic_negative'}],'train',self.root)
        with self.assertRaises(ValueError):guard_rows([{**self.base,'audio_source_video_id':'b'}],'train',self.root)

    def test_same_source_and_byte_alias_cannot_cross_split(self):
        for val in [{**self.base,'split':'val'},{**self.base,'split':'val','source_video_id':'b','media_sha256':'same'}]:
            with self.assertRaises(ValueError):assert_isolated([{**self.base,'media_sha256':'same'},val])

    def test_separate_rows_update_both_heads_with_masked_joint_loss(self):
        rows=[{**self.base,'corr_target':1.},
              {**self.base,'source_video_id':'c','kind':'Utility','corr_target':None,'corr_weight':0.,'utility_target':-.3},
              {**self.base,'source_video_id':'d','corr_target':None,'corr_weight':0.}]
        labels=make_labels(rows)
        self.assertEqual(int(labels.corr_mask.sum()),1);self.assertEqual(int(labels.utility_mask.sum()),1)
        model=DualHeadGate(7,5,8)
        features=GateFeatures(torch.randn(3,14),torch.randn(3,10),torch.zeros(3,4),torch.ones(3,dtype=torch.bool))
        loss,info=dual_loss(model.forward_features(features),labels);loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.correspondence_head.weight.grad.abs().sum()),0)
        self.assertGreater(float(model.utility_head.weight.grad.abs().sum()),0)
        self.assertEqual(info['corr_active'],1);self.assertEqual(info['utility_active'],1)

    def test_invalid_availability_does_not_create_negative(self):
        labels=make_labels([{**self.base,'corr_target':None,'quality_valid':False}])
        self.assertFalse(labels.corr_mask.any());self.assertTrue(torch.isnan(labels.corr_target).all())


class ProxyPolicy(unittest.TestCase):
    def test_independent_two_head_threshold_and_missing_mask(self):
        rule={'mode':'dual','fit_role':'val','selection_corr_threshold':0.,'selection_utility_threshold':0.}
        decision=apply_rule([1.,-1.,1.],[1.,1.,-1.],[True,True,True],rule)
        self.assertEqual(decision.tolist(),[True,False,False])
        self.assertFalse(apply_rule([1.],[1.],[False],rule).any())

    def test_no_test_fitted_policy_and_no_nonfinite_scores(self):
        rule={'mode':'utility','fit_role':'test','utility_threshold':0.}
        with self.assertRaises(ValueError):apply_rule([1.],[1.],[True],rule)
        with self.assertRaises(ValueError):apply_rule([1.],[np.nan],[True],{**rule,'fit_role':'val'})

    def test_constant_controls_encoded_without_json_infinity(self):
        rule={'mode':'corr','fit_role':'val','corr_threshold':'inf'}
        self.assertFalse(apply_rule([1.],[1.],[True],rule).any())
        self.assertTrue(apply_rule([1.],[1.],[True],{**rule,'corr_threshold':'-inf'}).all())


if __name__=='__main__':unittest.main()
