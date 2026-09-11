import unittest

import numpy as np

from scripts.prepare_partial_mcqa_datasets import answer_letter, parse_timestamp
from streamingtom.train_adbt_videomme import causal_prefix_length, rows_end_time


class PartialMCQADataTest(unittest.TestCase):
    def test_timestamp_parser_supports_benchmark_formats(self):
        self.assertEqual(parse_timestamp("09:11"), 551.0)
        self.assertEqual(parse_timestamp("00:09:11"), 551.0)

    def test_answer_is_normalized_to_candidate_letter(self):
        self.assertEqual(answer_letter(["one", "Two", "three"], " two "), "B")

    def test_query_uses_only_sampled_causal_prefix(self):
        timestamps = np.asarray([0.0, 2.0, 4.0, 6.0])
        self.assertEqual(causal_prefix_length(timestamps, {"query_time_seconds": 4}), 3)
        self.assertEqual(causal_prefix_length(timestamps, {"query_time_seconds": 1}), 1)
        self.assertEqual(causal_prefix_length(timestamps, {}), 4)

    def test_video_decode_boundary_is_latest_question(self):
        rows = [
            {"query_time_seconds": 3.0},
            {"query_time_seconds": 9.0},
            {"query_time_seconds": 5.0},
        ]
        self.assertEqual(rows_end_time(rows), 9.0)
        self.assertIsNone(rows_end_time([{"query_time_seconds": None}]))


if __name__ == "__main__":
    unittest.main()
