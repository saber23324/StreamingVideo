import copy
import unittest

import torch

from utility_gate.grounding.avqa_group_eval import interval, prefix_features, random_masks, validate_sources
from utility_gate.grounding.stats import mean_ci


class GroundingAVQATests(unittest.TestCase):
    def test_relation_answer_and_question_never_change_gate_tensors(self):
        torch.manual_seed(7)
        payload = {'visual': torch.randn(3, 196, 8), 'causal_audio': torch.randn(3, 4, 5),
                   'timestamps': torch.tensor([2., 4., 6.]), 'audio_valid': torch.tensor([True, False, True])}
        first = prefix_features(payload)
        second = prefix_features({**payload, 'question': 'future query', 'answer': 'C', 'question_modality': 'Sound'})
        for name in ('visual', 'audio', 'context', 'valid_audio'):
            torch.testing.assert_close(getattr(first, name), getattr(second, name))

    def test_random_matches_available_count_and_never_activates_absent_audio(self):
        policies = [
            {'question_id': 'a', 'available': [True, False, True], 'times': [2., 4., 6.], 'prefix_accepted': True, 'stream_selected': [True, False, True]},
            {'question_id': 'b', 'available': [True, True, False], 'times': [2., 4., 6.], 'prefix_accepted': False, 'stream_selected': [False, True, False]},
            {'question_id': 'c', 'available': [False, False, False], 'times': [2., 4., 6.], 'prefix_accepted': False, 'stream_selected': [False, False, False]}]
        result = random_masks(policies)
        self.assertEqual(sum(result['prefix'].values()), 1)
        self.assertFalse(result['prefix']['c'])
        for j in range(3):
            self.assertEqual(sum(v[j] for v in result['stream'].values()), sum(r['stream_selected'][j] for r in policies))
            for row in policies:
                self.assertFalse(result['stream'][row['question_id']][j] and not row['available'][j])

    def test_future_random_control_state_cannot_change_current_time_choices(self):
        rows = [{'question_id': str(i), 'available': [True, True], 'times': [2., 4.],
                 'prefix_accepted': i == 0, 'stream_selected': [i == 0, i == 0]} for i in range(6)]
        first = random_masks(rows)
        changed = copy.deepcopy(rows)
        for r in changed:
            r['stream_selected'][1] = False; r['available'][1] = False
        second = random_masks(changed)
        self.assertEqual([v[0] for v in first['stream'].values()], [v[0] for v in second['stream'].values()])

    def test_old_fit_evaluation_vggss_and_wrong_source_role_rejected(self):
        row = {'question_id': 'x', 'source_video_id': 'youtube:ABCDEFGHIJK', 'official_split': 'evaluation',
               'split': 'test', 'video_path': '/tmp/grounding_unique.mp4', 'media_sha256': 'abc'}
        validate_sources([row], set(), set(), set(), set(), set())
        for index in range(5):
            excluded = [set() for _ in range(5)]; excluded[index] = {'ABCDEFGHIJK'}
            with self.assertRaises(ValueError): validate_sources([row], *excluded)
        with self.assertRaises(ValueError): validate_sources([{**row, 'official_split': 'train'}], set(), set(), set(), set(), set())

    def test_one_source_has_no_population_interval_and_cluster_unit_is_source(self):
        value = interval([True], ['a'])
        self.assertIsNone(value['ci95'])
        self.assertEqual(value['sources'], 1)
        clustered = mean_ci([0., 0., 1., 1.], ['a', 'a', 'b', 'b'])
        self.assertEqual(clustered['n'], 4); self.assertEqual(clustered['sources'], 2)


if __name__ == '__main__': unittest.main()
