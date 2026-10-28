"""Program checks on SYNTHETIC features, never correspondence experiment results."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import wave
import subprocess
import sys

import numpy as np
import torch

from relevance_gate.experiment import train, infer, load_inputs
from relevance_gate.manifest import file_sha256, write_json, write_jsonl
from relevance_gate.media import decode_audio, frame_at_or_before
from relevance_gate.cache_reuse import reuse_cache_index
from relevance_gate.feature_cache import PairReader
from relevance_gate.manifest import cache_key


class FrozenCacheReuseTest(unittest.TestCase):
    def fixture(self, root):
        from PIL import Image
        image, audio = root / "image.png", root / "audio.wav"
        Image.new("RGB", (8, 8)).save(image)
        audio.write_bytes(b"synthetic media identity; not decoded")
        row = {"pair_id": "synthetic", "source_video_id": "synthetic", "audio_source_video_id": "synthetic",
               "split": "val", "frame_timestamp_s": 2., "audio_start_s": 0., "audio_end_s": 2.,
               "frame_path": str(image), "audio_path": str(audio), "visible_correspondence": None,
               "negative_type": "positive", "label_status": "pending", "provenance": "SYNTHETIC unit fixture"}
        metadata = {"kind": "imagebind", "scope": "SYNTHETIC"}
        key = cache_key(row, metadata, PairReader().checksums(row))
        tensor = root / "features.pt"
        torch.save({"visual": torch.ones(1, 4), "audio": torch.ones(1, 4), "cache_key": key,
                    "valid_aux": True, "quality_reason": "valid", "encoder_metadata": metadata}, tensor)
        index = root / "cache.json"
        write_json(index, {"manifest_sha256": "old-unlabeled-manifest", "encoder_metadata": metadata,
                           "entries": [{"pair_id": row["pair_id"], "key": key, "path": str(tensor)}]})
        return row, index

    def test_label_review_reuses_frozen_tensors_but_new_media_bytes_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row, index = self.fixture(root)
            row.update(visible_correspondence=1, label_status="audited", label_provenance="SYNTHETIC")
            manifest = root / "reviewed.jsonl"
            write_jsonl(manifest, [row])
            result = reuse_cache_index([row], index, root / "reviewed-cache.json", manifest, "imagebind")
            self.assertEqual(result["manifest_sha256"], file_sha256(manifest))
            self.assertEqual(result["entries"], json.loads(index.read_text())["entries"])
            Path(row["audio_path"]).write_bytes(b"changed media")
            with self.assertRaisesRegex(ValueError, "identity changed"):
                reuse_cache_index([row], index, root / "bad-cache.json", manifest)
            self.assertFalse((root / "bad-cache.json").exists())

    def test_changed_time_or_encoder_cannot_reuse_and_index_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row, index = self.fixture(root)
            manifest = root / "pairs.jsonl"
            write_jsonl(manifest, [row])
            with self.assertRaisesRegex(ValueError, "encoder"):
                reuse_cache_index([row], index, root / "wrong-encoder.json", manifest, "native")
            with self.assertRaises(FileExistsError):
                reuse_cache_index([row], index, index, manifest)
            row["audio_end_s"] = 1.9
            with self.assertRaisesRegex(ValueError, "identity changed"):
                reuse_cache_index([row], index, root / "wrong-time.json", manifest)

    def test_unresolved_debug_labels_cannot_enter_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row, index = self.fixture(root)
            row.update(split="debug", scope="development_only", visible_correspondence=1,
                       label_status="audited", label_provenance="SYNTHETIC fixture")
            manifest = root / "pairs.jsonl"
            write_jsonl(manifest, [row])
            payload = json.loads(index.read_text())
            payload["manifest_sha256"] = file_sha256(manifest)
            write_json(index, payload)
            with self.assertRaisesRegex(ValueError, "raw diagnostics only"):
                infer(argparse.Namespace(manifest=manifest, cache_index=index, split="all", raw_only=False))

    def test_nonfinite_cached_vectors_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row, index = self.fixture(root)
            manifest = root / "pairs.jsonl"
            write_jsonl(manifest, [row])
            path = Path(json.loads(index.read_text())["entries"][0]["path"])
            payload = torch.load(path, weights_only=True)
            payload["audio"][0, 0] = float("nan")
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "finite and frozen"):
                reuse_cache_index([row], index, root / "bad-cache.json", manifest)

    def test_g0_only_runner_calibrates_and_evaluates_without_train_or_native_encoder(self):
        # Real CLI orchestration on tiny SYNTHETIC vectors; not an AV result.
        with tempfile.TemporaryDirectory() as directory:
            from PIL import Image
            root = Path(directory)
            metadata = {"kind": "imagebind", "scope": "SYNTHETIC"}
            rows, entries = [], []
            for split in ("val", "test"):
                for label in (0, 1):
                    identifier = f"synthetic_{split}_{label}"
                    image, audio = root / (identifier + ".png"), root / (identifier + ".wav")
                    Image.new("RGB", (8, 8), (label, 0, 0)).save(image)
                    audio.write_bytes(identifier.encode())
                    row = {"pair_id": identifier, "source_video_id": identifier, "audio_source_video_id": identifier,
                           "split": split, "frame_timestamp_s": 2., "audio_start_s": 0., "audio_end_s": 2.,
                           "frame_path": str(image), "audio_path": str(audio), "visible_correspondence": label,
                           "negative_type": "positive" if label else "same_class", "label_status": "audited",
                           "label_provenance": "SYNTHETIC program fixture", "provenance": "unit test only"}
                    rows.append(row)
                    key = cache_key(row, metadata, PairReader().checksums(row))
                    tensor = root / (identifier + ".pt")
                    torch.save({"visual": torch.tensor([[1., 0.]]), "audio": torch.tensor([[float(label), float(1-label)]]),
                                "cache_key": key, "encoder_metadata": metadata, "valid_aux": True, "quality_reason": "valid"}, tensor)
                    entries.append({"pair_id": identifier, "key": key, "path": str(tensor)})
            manifest, index = root / "pairs.jsonl", root / "cache.json"
            write_jsonl(manifest, rows)
            write_json(index, {"manifest_sha256": file_sha256(manifest), "encoder_metadata": metadata, "entries": entries})
            repo = Path(__file__).resolve().parents[1]
            result = subprocess.run([sys.executable, "scripts/run_relevance_gate_m1.py", "--stage", "g0",
                                     "--manifest", str(manifest), "--work-dir", str(root / "run"),
                                     "--imagebind-index", str(index), "--legacy-id-files"], cwd=repo,
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((root / "run/results.json").read_text())
            self.assertEqual(report["learned_gates"], [])
            self.assertEqual(report["executed_stage"], "g0")
            self.assertFalse(report["M1_accepted"])
            self.assertTrue((root / "run/g0.test.json").is_file())
            self.assertFalse((root / "run/native_cache.json").exists())


class TinyTrainingPipelineTest(unittest.TestCase):
    def test_three_seed_training_calibration_and_test_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, entries = [], []
            generator = torch.Generator().manual_seed(123)
            for split, count in (("train", 16), ("val", 12), ("test", 12)):
                for index in range(count):
                    identifier = f"synthetic_{split}_{index}"
                    label = index % 2
                    row = {"pair_id": identifier, "source_video_id": identifier, "audio_source_video_id": identifier,
                           "split": split, "frame_timestamp_s": 2., "audio_start_s": 0., "audio_end_s": 2.,
                           "frame_path": "unit_fixture_only.png", "visible_correspondence": label,
                           "negative_type": "positive" if label else "cross_video", "label_status": "audited",
                           "label_provenance": "SYNTHETIC program fixture; no real AV semantics", "provenance": "unit test"}
                    rows.append(row)
                    path = root / (identifier + ".pt")
                    torch.save({"visual": torch.randn(1, 12, generator=generator),
                                "audio": torch.randn(3, 8, generator=generator), "valid_aux": True,
                                "quality_reason": "synthetic_unit_fixture", "cache_key": identifier}, path)
                    entries.append({"pair_id": identifier, "path": str(path), "key": identifier})
            manifest, cache_index = root / "manifest.jsonl", root / "cache.json"
            write_jsonl(manifest, rows)
            write_json(cache_index, {"manifest_sha256": file_sha256(manifest), "encoder_metadata": {"kind": "native", "scope": "SYNTHETIC"}, "entries": entries})
            args = argparse.Namespace(manifest=manifest, cache_index=cache_index, output_dir=root / "run", kind="g1",
                                      hidden=8, dropout=0., seeds="1,2,3", device="cpu", batch_size=8,
                                      epochs=2, learning_rate=.01, target_tpr=.95)
            reports = train(args)
            self.assertEqual(len(reports), 3)
            for seed in (1, 2, 3):
                checkpoint = root / "run" / f"g1_seed{seed}.pt"
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                self.assertNotIn("audio_bottleneck", payload)
                self.assertEqual(payload["seed"], seed)
                self.assertTrue(checkpoint.with_suffix(".calibration.json").is_file())
            checkpoint = root / "run/g1_seed1.pt"
            eval_args = argparse.Namespace(manifest=manifest, cache_index=cache_index, kind="g1", checkpoint=checkpoint,
                                           calibration=checkpoint.with_suffix(".calibration.json"), split="test", command="evaluate",
                                           device="cpu", batch_size=8, raw_only=False, bootstrap=0, output=root / "test.json")
            result = infer(eval_args)
            self.assertEqual(result["known_pairs"], 12)
            self.assertTrue((root / "test.predictions.jsonl").is_file())
            self.assertTrue((root / "test.failures.jsonl").is_file())
            self.assertTrue((root / "test_plots/reliability.png").is_file())
            # Mutating the manifest must invalidate this feature-index binding.
            manifest.write_text(manifest.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "different manifest"):
                load_inputs(manifest, cache_index)


class MediaPTSTest(unittest.TestCase):
    def test_future_native_samples_cannot_leak_through_resampling(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs = []
            for suffix in (0, 1):
                signal = (np.sin(np.arange(8000) / 8.) * 10000).astype(np.int16)
                if suffix:
                    signal[4000:] = -30000
                path = Path(directory) / f"audio{suffix}.wav"
                with wave.open(str(path), "wb") as output:
                    output.setnchannels(1); output.setsampwidth(2); output.setframerate(8000)
                    output.writeframes(signal.tobytes())
                values, _, details = decode_audio(path, sample_rate=16000, end_s=.5)
                self.assertTrue(details["truncate_before_resampling"])
                outputs.append(values)
            self.assertTrue(torch.equal(*outputs))

    def test_resampling_preserves_audio_elapsed_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            signal = (np.sin(np.arange(8000) / 8.) * 10000).astype(np.int16)
            with wave.open(str(path), "wb") as output:
                output.setnchannels(1); output.setsampwidth(2); output.setframerate(8000)
                output.writeframes(signal.tobytes())
            values, origin, diagnostics = decode_audio(path, sample_rate=16000)
            self.assertAlmostEqual(origin, 0.)
            self.assertLessEqual(abs(len(values) - 16000), 1)
            self.assertEqual(diagnostics["pts_gaps"], 0)
            self.assertTrue(torch.isfinite(values).all())

    def test_frame_selection_never_reads_next_frame(self):
        import av
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.mp4"
            with av.open(str(path), "w") as output:
                stream = output.add_stream("mpeg4", rate=5)
                stream.width = stream.height = 64; stream.pix_fmt = "yuv420p"
                for index in range(5):
                    frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), 40 * index, dtype=np.uint8), format="rgb24")
                    for packet in stream.encode(frame): output.mux(packet)
                for packet in stream.encode(): output.mux(packet)
            _, pts = frame_at_or_before(path, .59)
            self.assertAlmostEqual(pts, .4)
            self.assertLessEqual(pts, .59)


if __name__ == "__main__":
    unittest.main()
