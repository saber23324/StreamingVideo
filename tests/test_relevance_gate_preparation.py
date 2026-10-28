"""Media/protocol safeguards on synthetic fixtures, not AV experiment metrics."""
import copy
from fractions import Fraction
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import av
import cv2
import numpy as np
from PIL import Image
import torch

from relevance_gate.calibration import calibrate
from relevance_gate.data_tools import negative_candidates, freeze_reference_holdout
from relevance_gate.manifest import audit_manifest
from relevance_gate.official_media import align_official_frames
from relevance_gate.protocols import TRUSTED, WEAK, target_rows
from relevance_gate.feature_cache import PairReader
from relevance_gate.localization import native_mask_target, localization_metrics, attach_local_target
from scripts.download_avsbench_media import safe_member_path, extract_asset
from scripts.run_relevance_gate_m1 import readiness
from scripts.export_gate_manual_review import select_review_rows, select_quality_followups


def row(identifier="fixture", split="train", label=1):
    return {"pair_id": identifier, "source_video_id": identifier, "audio_source_video_id": identifier,
            "split": split, "frame_timestamp_s": 4., "audio_start_s": 2., "audio_end_s": 4.,
            "frame_path": "fixture.png", "audio_path": identifier + ".wav", "visible_correspondence": label,
            "label_status": "official" if label is not None else "pending", "label_provenance": "SYNTHETIC fixture" if label is not None else None,
            "negative_type": "positive", "provenance": "SYNTHETIC unit fixture", "category": "dog",
            "pairing_target": 1, "pairing_target_provenance": "original fixture pairing"}


class PairingProtocolTest(unittest.TestCase):
    def test_source_quality_followups_are_separate_and_not_repeated_per_frame(self):
        first, second, third = row("a"), row("b"), row("c")
        second["source_video_id"] = "a"
        audit = {"correlation_below_095": ["a", "c"], "checks": {}}
        result = select_quality_followups([first, second, third], audit, [])
        self.assertEqual([r["pair_id"] for r in result], ["a", "c"])
        result = select_quality_followups([first, second, third], audit, [third])
        self.assertEqual([r["pair_id"] for r in result], ["a"])

    def test_training_expansion_keeps_holdout_donors_and_labels_exact(self):
        reference = [row("val", "val"), row("test", "test")]
        reference[1].update(audio_source_video_id="test_donor", audio_path="test_donor.wav")
        expanded = [row("newtrain"), row("differentval", "val"), row("differenttest", "test")]
        result = freeze_reference_holdout(expanded, reference)
        self.assertEqual(result[1:], reference)
        leaking = row("train_alias", "train")
        leaking["source_video_id"] = "val"
        with self.assertRaisesRegex(ValueError, "leakage"):
            freeze_reference_holdout([leaking], reference)

    def test_final_review_selection_is_deterministic_and_ignores_model_scores(self):
        rows = []
        for split in ("train", "val", "test"):
            for i in range(30):
                candidate = row(f"{split}_{i}", split, None)
                candidate.update(negative_type="same_class", raw_score=float(i))
                rows.append(candidate)
        first = [r["pair_id"] for r in select_review_rows(rows, 1234)]
        changed = copy.deepcopy(rows)
        for r in changed: r["raw_score"] *= -10
        self.assertEqual(first, [r["pair_id"] for r in select_review_rows(changed, 1234)])
        self.assertEqual(len(first), 60)
        self.assertEqual(len(first), len(set(first)))

    def test_weak_targets_never_mutate_or_promote_semantic_unknowns(self):
        candidate = row(label=None)
        candidate.update(pairing_target=0, pairing_target_provenance="synthetic different-source pairing")
        original = copy.deepcopy(candidate)
        weak = target_rows([candidate], WEAK)[0]
        self.assertEqual(candidate, original)
        self.assertIsNone(weak["semantic_visible_correspondence"])
        self.assertEqual(weak["visible_correspondence"], 0)
        self.assertEqual(weak["label_status"], "weak_pairing_only")
        with self.assertRaisesRegex(ValueError, "ground truth"):
            audit_manifest([weak])

    def test_cross_video_targets_are_weak_and_other_controls_have_no_automatic_target(self):
        rows = negative_candidates([row("a"), row("b")])
        for candidate in rows[2:]:
            self.assertIsNone(candidate["visible_correspondence"])
            self.assertEqual(candidate["pairing_target"], 0 if candidate["negative_type"] == "cross_video" else None)

    def test_bad_target_or_missing_provenance_is_rejected(self):
        for target, provenance in ((True, "bad boolean"), (2, "bad target"), (0, None)):
            candidate = row(label=None)
            candidate.update(pairing_target=target, pairing_target_provenance=provenance)
            with self.assertRaises(ValueError):
                target_rows([candidate], WEAK)

    def test_manual_review_can_be_deferred_without_claiming_trusted_hard_negatives(self):
        rows = []
        for split in ("val", "test"):
            positive, candidate = row(split+"positive", split), row(split+"candidate", split, None)
            candidate.update(pairing_target=0, pairing_target_provenance="SYNTHETIC cross-video candidate", negative_type="cross_video")
            rows.extend((positive, candidate))
        report = readiness(rows, "g0", WEAK)
        self.assertTrue(report["ready"])
        self.assertTrue(report["manual_review_deferred"])
        self.assertEqual(report["trusted_test_hard_negatives"], 0)
        self.assertFalse(readiness(rows, "g0")["ready"])

    def test_calibration_cannot_mix_weak_targets_and_semantic_targets(self):
        records = [{**row("a", "val", 1), "raw_score": .8, "valid_aux": True, "label_protocol": TRUSTED},
                   {**row("b", "val", 0), "raw_score": .2, "valid_aux": True, "label_protocol": WEAK}]
        with self.assertRaisesRegex(ValueError, "cannot mix"):
            calibrate(records, "SYNTHETIC")
        records[0]["label_protocol"] = WEAK
        result = calibrate(records, "SYNTHETIC")
        self.assertEqual(result["label_protocol"], WEAK)
        self.assertFalse(result["semantic_validation_complete"])


class OfficialMediaTest(unittest.TestCase):
    def test_reviewed_positive_without_region_mask_is_not_given_phantom_localization(self):
        with self.assertRaisesRegex(ValueError, "no defensible"):
            attach_local_target(row(label=1), {})

    def test_source_mask_geometry_must_match_the_verified_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (8,8)).save(root/"frame.png")
            Image.new("L", (4,4), 255).save(root/"mask.png")
            candidate = {**row(), "frame_path": str(root/"frame.png"), "source_region_mask": str(root/"mask.png")}
            with self.assertRaisesRegex(ValueError, "geometry differ"):
                attach_local_target(candidate, {})

    def test_overlapping_audio_pts_become_explicit_quality_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            Image.new("RGB", (8,8)).save(path)
            candidate = row(label=None)
            candidate["frame_path"] = str(path)
            with patch("relevance_gate.feature_cache.decode_audio", side_effect=ValueError("overlapping/nonmonotonic audio PTS")):
                _, window, timing = PairReader()(candidate)
            self.assertFalse(window.valid)
            self.assertEqual(window.reason, "audio_pts_overlap")
            self.assertTrue(torch.isfinite(window.samples).all())
            self.assertEqual(timing["reason"], "audio_pts_overlap")

    def test_native_mask_mapping_matches_valid_patch_and_bilinear_layout(self):
        metadata = {"kind": "native", "pretrained": "lmms-lab/llava-onevision-qwen2-7b-ov",
                    "processor": {"class": "SigLipImageProcessor", "size": [384,384], "applied_crop": False}}
        target = native_mask_target(torch.ones(224,224), metadata)
        self.assertEqual(target.shape, (196,))
        torch.testing.assert_close(target, torch.ones(196))
        border = torch.zeros(384,384); border[:,378:] = 1
        with self.assertRaisesRegex(ValueError, "no support"):
            native_mask_target(border, metadata)
        with self.assertRaisesRegex(ValueError, "without crop"):
            native_mask_target(torch.ones(224,224), {**metadata, "processor": {**metadata["processor"], "applied_crop": True}})

    def test_localization_ranking_is_reported_separately_from_global_matching(self):
        target, scores = [1.]*10+[0.]*186, list(range(196,0,-1))
        scores = (torch.tensor(scores).float()/sum(scores)).tolist()
        records = [{"visible_correspondence": 1, "semantic_visible_correspondence": 1, "valid_aux": True,
                    "local_token_target": target, "conditional_local_scores": scores}]
        result = localization_metrics(records, budget=10)
        self.assertEqual(result["area_weighted_token_recall"], 1.)
        self.assertIn("no hard-pruning", result["scope"])

    def test_official_png_requires_pixel_verified_decoded_pts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "synthetic.mp4"
            with av.open(str(video), "w") as container:
                stream = container.add_stream("mpeg4", rate=16)
                stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
                for i in range(32):
                    pixels = np.full((32, 32, 3), i*7, dtype=np.uint8)
                    frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                    frame.pts, frame.time_base = i, Fraction(1, 16)
                    for packet in stream.encode(frame): container.mux(packet)
                for packet in stream.encode(): container.mux(packet)
            with av.open(str(video)) as container:
                decoded = list(container.decode(video=0))
            paths = {}
            for index, decoded_index in ((1, 15), (2, 31)):
                path = root / f"official_{index}.png"
                Image.fromarray(cv2.resize(decoded[decoded_index].to_ndarray(format="rgb24"), (224, 224))).save(path)
                paths[index] = path
            result = align_official_frames(video, paths)
            self.assertEqual([r["timestamp_s"] for r in result], [15/16, 31/16])
            self.assertEqual([r["pixel_mae"] for r in result], [0., 0.])
            Image.new("RGB", (224, 224), "red").save(paths[1])
            with self.assertRaisesRegex(ValueError, "pixel check failed"):
                align_official_frames(video, paths)

    def test_archive_extraction_rejects_traversal_and_conflicting_files(self):
        for path in ("../escape", "/absolute", "data/../../escape", "data\\escape"):
            with self.assertRaises(ValueError): safe_member_path(path)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "synthetic.tar.gz"
            with tarfile.open(archive, "w:gz") as output:
                value = b"synthetic media"
                info = tarfile.TarInfo("s4_data/raw_videos/train/dog/fixture.mp4")
                info.size = len(value)
                output.addfile(info, io.BytesIO(value))
            extract_asset("s4", archive, root)
            target = root / "Single-source/s4_data/raw_videos/train/dog/fixture.mp4"
            self.assertEqual(target.read_bytes(), b"synthetic media")
            extract_asset("s4", archive, root)
            target.write_bytes(b"existing user bytes")
            with self.assertRaisesRegex(ValueError, "differs"):
                extract_asset("s4", archive, root)
            self.assertEqual(target.read_bytes(), b"existing user bytes")


if __name__ == "__main__":
    unittest.main()
