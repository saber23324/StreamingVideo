from pathlib import Path
import tempfile
import unittest

from utility_gate.grounding.llp_parser import clip_identity, labels_at, read_intervals, subtype
from relevance_gate.media import causal_window
import torch


class LLPLabelTests(unittest.TestCase):
    def test_all_five_subtypes(self):
        self.assertEqual(subtype({'Dog'}, {'Dog'}), 'joint_same_class')
        self.assertEqual(subtype({'Dog'}, {'Cat'}), 'different_categories')
        self.assertEqual(subtype({'Dog'}, set()), 'audio_only')
        self.assertEqual(subtype(set(), {'Dog'}), 'visual_only')
        self.assertEqual(subtype(set(), set()), 'neither')

    def test_window_does_not_invent_cross_second_joint_event(self):
        value = labels_at([(0, 1, 'Dog')], [(1, 2, 'Dog')], 2)
        self.assertFalse(value['current_bin_joint'])
        self.assertFalse(value['causal_window_joint'])
        self.assertTrue(labels_at([(0, 1, 'Dog')], [(0, 1, 'Dog')], 2)['causal_window_joint'])

    def test_half_open_boundaries_and_first_second(self):
        a, v = [(0, 2, 'Dog'), (2, 4, 'Cat')], [(0, 2, 'Dog')]
        self.assertTrue(labels_at(a, v, 2)['current_bin_joint'])
        self.assertFalse(labels_at(a, v, 3)['current_bin_joint'])
        self.assertEqual(labels_at(a, v, 1)['causal_label_window'], [0, 1])

    def test_dense_schema_and_id(self):
        self.assertEqual(clip_identity('KSRjje7GH44_60_70'), ('KSRjje7GH44', 60, 70))
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'labels.csv'
            p.write_text('filename\tonset\toffset\tevent_labels\nKSRjje7GH44_60_70\t0\t8\tSpeech\n')
            self.assertEqual(read_intervals(p)['KSRjje7GH44_60_70'], [(0, 8, 'Speech')])
            p.write_text('filename\tevent_labels\nKSRjje7GH44_60_70\tSpeech\n')
            with self.assertRaises(ValueError): read_intervals(p)

    def test_actual_waveform_windows_exclude_future_and_old_values(self):
        wave = torch.arange(80000, dtype=torch.float32)
        for t, first, last in [(2, 0, 32000), (4, 32000, 64000)]:
            value = causal_window(wave, t)
            torch.testing.assert_close(value.samples, wave[first:last])
            changed = wave.clone(); changed[last:] = -1e10
            torch.testing.assert_close(causal_window(changed, t).samples, value.samples)
        self.assertEqual(causal_window(wave, float('nan')).reason, 'invalid_timestamp')

    def test_bad_official_intervals_are_logged_and_never_repaired(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'labels.csv'
            p.write_text('filename\tonset\toffset\tevent_labels\nKSRjje7GH44_60_70\t9\t0\tSpeech\n')
            with self.assertRaises(ValueError): read_intervals(p)
            rejected = []
            self.assertEqual(read_intervals(p, strict=False, rejections=rejected), {})
            self.assertEqual(len(rejected), 1)
            self.assertEqual(rejected[0]['onset'], '9')


if __name__ == '__main__': unittest.main()
