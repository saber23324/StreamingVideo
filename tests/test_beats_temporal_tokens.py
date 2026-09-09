import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from streamingtom.audio_bottleneck.beats_encoder import BEATsAudioEncoder


class FakeBEATs(nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor = object()

    def extract_features(self, batch, padding_mask=None):
        batch_size = batch.shape[0]
        representation = torch.arange(
            batch_size * 5 * 12, device=batch.device, dtype=torch.float32
        ).reshape(batch_size, 5, 12)
        return representation, None


def make_encoder(output_mode: str) -> BEATsAudioEncoder:
    encoder = BEATsAudioEncoder.__new__(BEATsAudioEncoder)
    nn.Module.__init__(encoder)
    encoder.output_mode = output_mode
    encoder.batch_size = 2
    encoder.output_dim = 12
    encoder.beats = FakeBEATs()
    return encoder


class BEATsTemporalTokensTest(unittest.TestCase):
    def test_phase4_preserves_temporal_axis(self):
        windows = torch.randn(3, 32000)
        output = make_encoder("temporal").encode_windows(
            windows, device=torch.device("cpu")
        )

        self.assertEqual(output.shape, (3, 5, 12))
        self.assertGreater((output[:, 1:] - output[:, :-1]).abs().sum(), 0)

    def test_legacy_mean_mode_remains_reproducible(self):
        windows = torch.randn(3, 32000)
        output = make_encoder("mean").encode_windows(
            windows, device=torch.device("cpu")
        )

        self.assertEqual(output.shape, (3, 12))

    def test_crossvideo_ablation_decodes_explicit_wrong_video(self):
        encoder = make_encoder("temporal")
        encoder.sample_rate = 16000
        encoder.window_seconds = 2.0
        encoder.source = "beats"
        wrong_waveform = torch.randn(64000)
        with patch.object(
            encoder, "decode_audio", return_value=wrong_waveform
        ) as decode_audio, patch.object(
            encoder,
            "encode_windows",
            return_value=torch.randn(2, 5, 12),
        ):
            result = encoder.encode_video(
                "current.mp4",
                [1.0, 2.0],
                device=torch.device("cpu"),
                ablation="crossvideo",
                crossvideo_path="wrong.mp4",
            )

        decode_audio.assert_called_once_with("wrong.mp4", 16000)
        self.assertEqual(result["audio_source_path"], "wrong.mp4")
        self.assertEqual(result["ablation"], "crossvideo")


if __name__ == "__main__":
    unittest.main()
