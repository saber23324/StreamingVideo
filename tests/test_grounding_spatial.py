import unittest
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from utility_gate.grounding.spatial_mapping import (ImageTransform, native_occupancy,
    pooled_occupancy, routing_metrics, token_centers, union_area)


class SpatialMappingTests(unittest.TestCase):
    def setUp(self):
        self.transform = ImageTransform(640, 360)
        self.uniform = np.full((64, 196), 1/196)

    def test_full_image_and_uniform_area(self):
        r = pooled_occupancy([[0, 0, 1, 1]], self.transform)
        np.testing.assert_allclose(r, 1)
        metrics = routing_metrics(self.uniform, r, [[0, 0, 1, 1]], self.transform)
        self.assertAlmostEqual(metrics['mass'], 1)
        self.assertAlmostEqual(metrics['excess_mass'], 0)

    def test_fractional_union_not_double_counted(self):
        boxes = [[0, 0, .6, .6], [.3, .3, .9, .9]]
        self.assertAlmostEqual(union_area(boxes), .63)
        once = pooled_occupancy(boxes, self.transform)
        twice = pooled_occupancy(boxes+boxes, self.transform)
        np.testing.assert_allclose(once, twice)
        result = routing_metrics(self.uniform, once, boxes, self.transform)
        self.assertAlmostEqual(result['mass'], result['area_baseline'])

    def test_valid_convolution_excludes_last_six_pixels(self):
        transform = ImageTransform(384, 384)
        r = pooled_occupancy([[378/384, 0, 1, 1]], transform)
        np.testing.assert_array_equal(r, 0)
        self.assertFalse(np.allclose(pooled_occupancy([[0, 0, .5, .5]], transform).mean(), .25))

    def test_partial_cell_is_continuous_area(self):
        value = native_occupancy([[0, 0, 7/384, 14/384]], ImageTransform(384, 384))
        self.assertAlmostEqual(value[0, 0], .5)
        self.assertEqual(np.count_nonzero(value), 1)

    def test_interpolation_order_and_centers_match_torch(self):
        grid = torch.arange(729, dtype=torch.float64).reshape(1, 1, 27, 27)
        actual = F.interpolate(grid, size=(14, 14), mode='bilinear', align_corners=False).flatten().numpy()
        centers = token_centers(ImageTransform(384, 384))
        expected = ((centers[:, 1]/14-.5)*27+(centers[:, 0]/14-.5))
        np.testing.assert_allclose(actual, expected)
        self.assertEqual(centers[1, 1], centers[0, 1])
        self.assertGreater(centers[14, 1], centers[0, 1])

    def test_empty_outside_and_nonfinite_boxes(self):
        np.testing.assert_array_equal(pooled_occupancy([], self.transform), 0)
        np.testing.assert_array_equal(pooled_occupancy([[2, 2, 3, 3]], self.transform), 0)
        np.testing.assert_allclose(pooled_occupancy([[-1, -1, 2, 2]], self.transform), 1)
        for box in ([[0, 0, float('nan'), 1]], [[.8, 0, .2, 1]]):
            with self.assertRaises(ValueError): pooled_occupancy(box, self.transform)

    def test_center_crop_letterbox_and_rectangular_warp(self):
        for mode in ['warp', 'center_crop', 'letterbox']:
            transform = ImageTransform(800, 400, mode)
            boxes = transform.boxes([[0, 0, 1, 1]])
            expected = [0, 0, 384, 384] if mode != 'letterbox' else [0, 96, 384, 288]
            np.testing.assert_allclose(boxes[0], expected)
        self.assertEqual(len(ImageTransform(800, 400, 'center_crop').boxes([[0, 0, .1, 1]])), 0)

    def test_routing_shape_mass_finiteness_and_pointing(self):
        r = pooled_occupancy([[0, 0, .5, .5]], self.transform)
        p = np.zeros((64, 196)); p[:, 0] = 1
        value = routing_metrics(p, r, [[0, 0, .5, .5]], self.transform)
        self.assertTrue(value['pointing'])
        self.assertAlmostEqual(value['mass'], 1)
        for invalid in [np.zeros((64, 196)), np.full((64, 196), np.nan), np.ones((64, 64))]:
            with self.assertRaises(ValueError): routing_metrics(invalid, r, [], self.transform)

    def test_future_decoded_frame_fails_before_encoding(self):
        from utility_gate.grounding.native import encode_times
        with patch('utility_gate.grounding.native.frame_at_or_before', return_value=(np.zeros((2, 2, 3)), 2.001)):
            with self.assertRaisesRegex(ValueError, 'future visual frame'):
                encode_times(None, 'synthetic-not-real-media', [2.])

    def test_annotation_frame_mismatch_is_logged_and_excluded(self):
        from utility_gate.grounding.vggss_eval import run
        fake = SimpleNamespace(device='cpu', audio_encoder=SimpleNamespace(
            encode_windows=lambda *args: torch.zeros(1, 96, 768)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'datasets').mkdir()
            row = {'clip_id': 'synthetic', 'source_video_id': 'synthetic', 'bbox': [[0, 0, 1, 1]],
                   'media_path': 'synthetic', 'annotation_tolerance_s': .1}
            (root/'datasets/vggss_manifest.jsonl').write_text(json.dumps(row)+'\n')
            with patch('utility_gate.grounding.vggss_eval.StrictStreamingRuntime', return_value=fake), \
                 patch('utility_gate.grounding.vggss_eval.lock_evaluation'), \
                 patch('utility_gate.grounding.vggss_eval.gates', return_value=(None, 0, None, 0)), \
                 patch('utility_gate.grounding.vggss_eval.encode_times', return_value=({'frame_pts_s': [2., 4., 4.8]}, [])), \
                 redirect_stdout(io.StringIO()):
                run(root)
            summary = json.loads((root/'spatial/summary.json').read_text())
            self.assertEqual(summary['spatial_valid'], 0)
            failure = json.loads((root/'spatial/failures.jsonl').read_text())
            self.assertIn('annotation_frame_pts_error', failure['reason'])


if __name__ == '__main__': unittest.main()
