import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from PIL import Image

from relevance_gate.calibration import calibrate
from relevance_gate.data_tools import avsbench_manifest, negative_candidates, source_id, merge_reviews, public_example_metadata
from relevance_gate.evaluation import classification_metrics, evaluate_records
from relevance_gate.manifest import audit_manifest, cache_key
from relevance_gate.media import causal_window
from relevance_gate.models import PooledAVGate, PrototypeAVGate, ImageBindCosineGate, masked_pool, predict
from relevance_gate.policy import RelevanceRoutingPolicy
from streamingtom.audio_bottleneck.adbt import AudioConditionedBottleneck
from scripts.run_relevance_gate_m1 import readiness


CAL = {"scale": 1., "bias": 0., "threshold": .5, "version": "unit-fixture-only"}


def pair(identifier="one", split="train", label=1, stratum="positive"):
    return {"pair_id": identifier, "source_video_id": identifier, "audio_source_video_id": identifier,
            "split": split, "frame_timestamp_s": 4., "audio_start_s": 2., "audio_end_s": 4.,
            "frame_path": "fixture.png", "audio_path": "fixture.wav", "category": "dog",
            "visible_correspondence": label, "negative_type": stratum,
            "label_status": "audited" if label is not None else "pending",
            "label_provenance": "synthetic unit fixture, not scientific data" if label is not None else None,
            "provenance": "unit test only"}


class GateModelsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.visual = torch.randn(3, 5, 12)
        self.audio = torch.randn(3, 7, 8)

    def test_pooled_global_shapes_and_no_localization_claim(self):
        for kind in ("g1", "g2"):
            model = PooledAVGate(12, 8, hidden=16, variant=kind).eval()
            logits, valid, local = model(self.visual, self.audio)
            self.assertEqual(logits.shape, (3,))
            self.assertTrue(valid.all())
            self.assertIsNone(local)

    def test_masked_padding_nan_does_not_poison_pool(self):
        values = torch.tensor([[[1., 3.], [float("nan"), float("inf")]]])
        mean, valid = masked_pool(values, torch.tensor([[True, False]]))
        torch.testing.assert_close(mean, torch.tensor([[1., 3.]]))
        self.assertTrue(valid.item())

    def test_unmasked_nan_invalidates_sample(self):
        self.audio[0, 0, 0] = float("nan")
        output = predict(PooledAVGate(12, 8).eval(), self.visual, self.audio, torch.ones(3, dtype=torch.bool), CAL)
        self.assertFalse(output.accepted[0])
        self.assertFalse(output.valid_aux[0])
        self.assertTrue(torch.isfinite(output.match_score).all())

    def test_all_missing_masks_reject(self):
        output = predict(PooledAVGate(12, 8).eval(), self.visual, self.audio, torch.ones(3, dtype=torch.bool), CAL,
                         audio_mask=torch.zeros(3, 7, dtype=torch.bool))
        self.assertFalse(output.accepted.any())
        self.assertEqual(output.match_score.sum(), 0.)

    def test_no_audio_tokens_reject_without_nan(self):
        output = predict(PooledAVGate(12, 8).eval(), self.visual, torch.empty(3, 0, 8), torch.ones(3, dtype=torch.bool), CAL)
        self.assertFalse(output.accepted.any())
        self.assertTrue(torch.isfinite(output.match_logit).all())

    def test_projection_and_mlp_train_but_encoder_features_detach(self):
        model = PooledAVGate(12, 8)
        visual = self.visual.requires_grad_()
        audio = self.audio.requires_grad_()
        before = {n: p.clone() for n, p in model.named_parameters()}
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        logits, _, _ = model(visual, audio)
        torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1., 0.])).backward()
        optimizer.step()
        self.assertIsNone(visual.grad); self.assertIsNone(audio.grad)
        self.assertTrue(any(not torch.equal(p, before[n]) for n, p in model.named_parameters()))
        self.assertIsNotNone(model.visual_projection.weight.grad)
        self.assertIsNotNone(model.audio_projection.weight.grad)

    def test_mask_shape_errors_are_explicit(self):
        with self.assertRaises(ValueError):
            masked_pool(self.visual, torch.ones(3, 4, dtype=torch.bool))
        with self.assertRaises(ValueError):
            masked_pool(self.visual, torch.ones(3, 5))

    def test_calibration_is_required_and_threshold_ties_accept(self):
        class Fixed(nn.Module):
            def forward(self, v, a, *args):
                return torch.zeros(3), torch.ones(3, dtype=torch.bool), None
        with self.assertRaises(ValueError):
            predict(Fixed(), self.visual, self.audio, torch.ones(3, dtype=torch.bool), None)
        out = predict(Fixed(), self.visual, self.audio, torch.tensor([True, False, True]), CAL)
        self.assertEqual(out.accepted.tolist(), [True, False, True])

    def test_imagebind_cosine_not_unaligned_beats_dot_product(self):
        gate = ImageBindCosineGate()
        raw, valid, local = gate(torch.eye(3), torch.eye(3))
        torch.testing.assert_close(raw, torch.ones(3))
        self.assertTrue(valid.all()); self.assertIsNone(local)
        with self.assertRaises(ValueError):
            gate(torch.ones(3, 12), torch.ones(3, 8))

    def test_zero_imagebind_vectors_invalid(self):
        _, valid, _ = ImageBindCosineGate()(torch.zeros(3, 8), torch.randn(3, 8))
        self.assertFalse(valid.any())

    def test_imagebind_output_labels_cosine_and_calibrated_logit_honestly(self):
        calibration = {**CAL, "scale": 2., "bias": -1.}
        out = predict(ImageBindCosineGate(), torch.eye(3), torch.eye(3), torch.ones(3, dtype=torch.bool), calibration)
        self.assertEqual(out.diagnostics["raw_score_kind"], "cosine_similarity")
        self.assertEqual(out.diagnostics["match_logit_kind"], "validation_calibrated_logit")
        torch.testing.assert_close(out.match_logit, torch.ones(3))

    def test_prototype_has_independent_null_classifier(self):
        model = PrototypeAVGate(12, 8, hidden=16).eval()
        with torch.no_grad():
            model.match_null_head[-1].weight.zero_()
            model.match_null_head[-1].bias.copy_(torch.tensor([20., -20.]))
        logits, valid, local = model(self.visual, self.audio)
        torch.testing.assert_close(local.sum(-1), torch.ones(3))
        output = predict(model, self.visual, self.audio, valid, CAL)
        self.assertFalse(output.accepted.any())
        self.assertEqual(torch.count_nonzero(output.local_token_scores), 0)
        self.assertTrue((logits < 0).all())

    def test_prototype_all_invalid_masks_and_inf_are_finite(self):
        model = PrototypeAVGate(12, 8, hidden=16).eval()
        self.visual[:] = float("inf")
        output = predict(model, self.visual, self.audio, torch.ones(3, dtype=torch.bool), CAL,
                         visual_mask=torch.zeros(3, 5, dtype=torch.bool))
        self.assertTrue(torch.isfinite(output.match_logit).all())
        self.assertFalse(output.accepted.any())


class CausalAudioTest(unittest.TestCase):
    def setUp(self):
        self.wave = torch.arange(1., 41.)

    def test_t0_is_invalid_empty_history(self):
        window = causal_window(self.wave, 0., sample_rate=10)
        self.assertFalse(window.valid); self.assertEqual(window.samples.sum(), 0)

    def test_partial_history_is_left_padded(self):
        window = causal_window(self.wave, 1., sample_rate=10)
        self.assertTrue(window.valid)
        torch.testing.assert_close(window.samples[:10], torch.zeros(10))
        torch.testing.assert_close(window.samples[10:], self.wave[:10])

    def test_future_samples_cannot_affect_window(self):
        first = causal_window(self.wave, 2.1, sample_rate=10)
        self.wave[21:] = 1e6
        second = causal_window(self.wave, 2.1, sample_rate=10)
        torch.testing.assert_close(first.samples, second.samples)
        self.assertEqual(first.samples[-1], 21.)

    def test_after_track_end_rejects_instead_of_reusing_stale_audio(self):
        window = causal_window(self.wave[:20], 5., sample_rate=10)
        self.assertFalse(window.valid); self.assertEqual(window.reason, "out_of_window")
        self.assertEqual(window.samples.sum(), 0)

    def test_missing_silence_malformed_nan_are_distinguished(self):
        inputs = ((None, "missing_audio"), (torch.zeros(40), "silence"),
                  (torch.ones(2, 40), "malformed_audio"), (torch.full((40,), float("nan")), "nonfinite_audio"))
        for wave, reason in inputs:
            window = causal_window(wave, 2., sample_rate=10)
            self.assertFalse(window.valid); self.assertEqual(window.reason, reason)
            self.assertTrue(torch.isfinite(window.samples).all())

    def test_stream_start_offset_is_preserved(self):
        window = causal_window(self.wave, 1.5, sample_rate=10, origin_s=1.)
        self.assertEqual(window.sample_mask.sum(), 5)
        torch.testing.assert_close(window.samples[-5:], self.wave[:5])

    def test_invalid_timestamps_and_length_never_access_future(self):
        for timestamp in (float("nan"), float("inf"), -1.):
            self.assertFalse(causal_window(self.wave, timestamp, sample_rate=10).valid)
        with self.assertRaises(ValueError):
            causal_window(self.wave, 1., sample_rate=0)


class ManifestTest(unittest.TestCase):
    def test_declared_source_checksum_is_checked_against_actual_video(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.mp4"
            path.write_bytes(b"SYNTHETIC fixture media bytes; not decoded")
            row = pair()
            row.pop("frame_path")
            row.pop("audio_path")
            row.update(video_path=str(path), source_media_sha256="stale-declaration")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                audit_manifest([row], check_files=True)

    def test_public_ms3_cross_split_source_is_debug_without_guessing_clip(self):
        source = {"id": "abcdefghijk", "url": "https://official.example/multi_sources/abcdefghijk.mp4"}
        official = [{"video_id": "abcdefghijk_0", "split": "train", "subset": "ms3"},
                    {"video_id": "abcdefghijk_1", "split": "val", "subset": "ms3"}]
        result = public_example_metadata(source, official)
        self.assertEqual(result["split"], "debug")
        self.assertEqual(result["split_provenance"], "official_source_crosses_splits")
        self.assertFalse(result["clip_id_matched"])
        self.assertEqual(result["clip_id"], source["id"])

    def test_public_source_group_can_match_split_without_claiming_clip_identity(self):
        source = {"id": "abcdefghijk", "url": "https://official.example/multi_sources/abcdefghijk.mp4"}
        official = [{"video_id": "abcdefghijk_0", "split": "train", "subset": "ms3"},
                    {"video_id": "abcdefghijk_1", "split": "train", "subset": "ms3"}]
        result = public_example_metadata(source, official)
        self.assertEqual(result["split"], "train")
        self.assertFalse(result["clip_id_matched"])
        self.assertEqual(len(result["official_metadata_candidates"]), 2)

    def test_official_metadata_overrides_old_inventory_default_test(self):
        source = {"id": "abcdefghij", "url": "https://official.example/single_source/abcdefghij.mp4",
                  "metadata_matches": [{"name": "unverified", "split": "test"}]}
        official = [{"name": "_abcdefghij", "category": "dog", "split": "train", "subset": "s4"}]
        result = public_example_metadata(source, official)
        self.assertEqual(result["source_video_id"], "_abcdefghij")
        self.assertEqual(result["split"], "train")
        self.assertTrue(result["clip_id_matched"])
        self.assertEqual(public_example_metadata(source, [])["split"], "debug")

    def test_debug_partition_requires_scope_and_cannot_be_promoted_implicitly(self):
        row = pair("a", "debug", None)
        with self.assertRaisesRegex(ValueError, "development"):
            audit_manifest([row])
        row.update(scope="development_only", split_provenance="official_source_crosses_splits")
        self.assertEqual(audit_manifest([row])["splits"], {"debug": 1})
        row["split"] = "test"
        with self.assertRaisesRegex(ValueError, "unresolved"):
            audit_manifest([row])

    def test_g0_readiness_does_not_require_learned_gate_training_labels(self):
        rows = [pair(f"{split}_{label}", split, label, "positive" if label else "same_class")
                for split in ("val", "test") for label in (0, 1)]
        self.assertTrue(readiness(rows, "g0")["ready"])
        self.assertFalse(readiness(rows, "learned")["ready"])
        self.assertFalse(readiness(rows)["ready"])
        rows[2].update(visible_correspondence=None, label_status="pending", label_provenance=None)
        self.assertFalse(readiness(rows, "g0")["ready"])

    def test_media_content_hashes_catch_duplicate_source_alias_without_declared_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "frame.png"
            Image.new("RGB", (8, 8)).save(image)
            rows = [pair("a", "train"), pair("b", "test")]
            for row in rows:
                media = root / (row["pair_id"] + ".wav")
                media.write_bytes(b"identical unit-fixture media bytes")
                row.update(frame_path=str(image), audio_path=str(media))
            with self.assertRaisesRegex(ValueError, "leakage"):
                audit_manifest(rows, check_files=True)

    def test_official_converter_keeps_empty_mask_unknown_and_requires_pts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            single = root / "Single-source"
            single.mkdir()
            (single / "s4_meta_data.csv").write_text("name,start,category,split\na,0,dog,train\nb,0,dog,val\n")
            times = []
            for video, split, foreground in (("a", "train", 255), ("b", "val", 0)):
                frame = single / "s4_data/visual_frames" / split / "dog" / video / f"{video}_1.png"
                mask = single / "gt_masks" / split / "dog" / video / f"{video}_1.png"
                audio = single / "s4_data/audio_wav" / split / "dog" / f"{video}.wav"
                for path in (frame, mask, audio): path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8)).save(frame)
                Image.new("L", (8, 8), foreground).save(mask)
                audio.write_bytes(video.encode())
                times.append({"video_id": video, "frame_index": 1, "timestamp_s": .9, "provenance": "synthetic decoded-PTS fixture"})
            rows, audit = avsbench_manifest(root, "s4", times)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["visible_correspondence"], 1)
            self.assertIsNone(rows[1]["visible_correspondence"])
            self.assertGreater(audit["invalid_counts"]["missing_verified_frame_timestamp"], 0)

    def test_labels_stay_unknown_for_generated_controls(self):
        rows = negative_candidates([pair("a"), pair("b")])
        for row in rows[2:]:
            self.assertIsNone(row["visible_correspondence"])
        self.assertGreater(audit_manifest(rows)["labels"]["unknown"], 0)

    def test_sources_on_both_pair_sides_cannot_cross_splits(self):
        a, b = pair("a"), pair("b", split="test")
        b["audio_source_video_id"] = "a"
        with self.assertRaisesRegex(ValueError, "leakage"):
            audit_manifest([a, b])

    def test_duplicate_content_under_new_id_cannot_leak(self):
        a, b = pair("a"), pair("b", split="test")
        a["source_media_sha256"] = b["source_media_sha256"] = "samehash"
        with self.assertRaisesRegex(ValueError, "leakage"):
            audit_manifest([a, b])

    def test_legacy_overlap_and_future_windows_fail(self):
        with self.assertRaisesRegex(ValueError, "protected"):
            audit_manifest([pair("a")], legacy_ids={"a"})
        row = pair(); row["audio_end_s"] = 5.
        with self.assertRaisesRegex(ValueError, "noncausal"):
            audit_manifest([row])

    def test_unreviewed_label_is_rejected(self):
        row = pair(); row["label_status"] = "pending"
        with self.assertRaises(ValueError):
            audit_manifest([row])

    def test_cache_identity_includes_timestamp_audio_and_checkpoint(self):
        row = pair(); original = cache_key(row, {"checkpoint": "a"}, {"media": "one"})
        changed = copy.deepcopy(row); changed["audio_start_s"] += .001
        self.assertNotEqual(original, cache_key(changed, {"checkpoint": "a"}, {"media": "one"}))
        self.assertNotEqual(original, cache_key(row, {"checkpoint": "b"}, {"media": "one"}))
        self.assertNotEqual(original, cache_key(row, {"checkpoint": "a"}, {"media": "two"}))

    def test_clip_suffix_is_grouped_without_truncating_yt_underscore(self):
        self.assertEqual(source_id("xv7eM6-UmkY_36"), "xv7eM6-UmkY")
        self.assertEqual(source_id("abc_defghij"), "abc_defghij")

    def test_review_requires_named_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.csv"
            path.write_text("pair_id,visible_correspondence,reviewer,evidence\none,0,,\n")
            with self.assertRaises(ValueError):
                merge_reviews([pair()], path)

    def test_blank_review_rows_preserve_unknown_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.csv"
            path.write_text("pair_id,visible_correspondence,reviewer,evidence\none,,,\n")
            rows = merge_reviews([pair(label=None)], path)
            self.assertIsNone(rows[0]["visible_correspondence"])


class CalibrationTest(unittest.TestCase):
    def test_bootstrap_clusters_connect_reused_audio_sources(self):
        rows = [{**pair("a", "test", 1), "match_score": .9, "accepted": True, "valid_aux": True},
                {**pair("b", "test", 0, "cross_video"), "match_score": .1, "accepted": False, "valid_aux": True}]
        rows[1]["audio_source_video_id"] = "a"
        result = evaluate_records(rows)
        self.assertEqual(result["independent_source_components"], 1)
        self.assertEqual(result["source_video_bootstrap_ci95"], {})
        self.assertEqual(result["by_stratum"]["same_class"]["total"], 0)

    def records(self, split="val"):
        return [{**pair(f"v{i}", split, i % 2, "cross_video" if i % 2 == 0 else "positive"),
                 "raw_score": float(i % 2), "valid_aux": True} for i in range(40)]

    def test_test_labels_cannot_fit_threshold(self):
        with self.assertRaisesRegex(ValueError, "validation"):
            calibrate(self.records("test"), "model")

    def test_frozen_calibration_is_deterministic(self):
        a, b = calibrate(self.records(), "model"), calibrate(self.records(), "model")
        self.assertEqual(a, b); self.assertTrue(a["target_attainable"])
        self.assertGreaterEqual(a["scale"], 0)

    def test_quality_limited_target_is_reported_unattainable(self):
        rows = self.records()
        rows[1]["valid_aux"] = False
        rows[3]["valid_aux"] = False
        result = calibrate(rows, "model")
        self.assertFalse(result["target_attainable"])
        self.assertAlmostEqual(result["max_quality_limited_tpr"], .9)
        self.assertGreater(result["threshold"], 0.)
        negatives = [r for r in rows if r["visible_correspondence"] == 0]
        from relevance_gate.calibration import probabilities
        scores = probabilities([r["raw_score"] for r in negatives], [r["valid_aux"] for r in negatives], result)
        self.assertFalse((scores >= result["threshold"]).any())

    def test_small_fpr95_is_not_reported_as_reliable(self):
        result = classification_metrics([0, 1], [.1, .9], [False, True])
        self.assertEqual(result["auroc"], 1.)
        self.assertIsNone(result["fpr_at_tpr95"])

    def test_tied_ranking_is_half_not_order_dependent(self):
        result = classification_metrics([0, 1, 0, 1], [.5] * 4, [True] * 4)
        self.assertEqual(result["auroc"], .5)
        self.assertEqual(result["false_accept_rate"], 1.)

    def test_probability_metrics_reject_raw_cosine(self):
        with self.assertRaises(ValueError):
            classification_metrics([0, 1], [-.5, .5], [False, True])

    def test_unknowns_and_invalid_examples_remain_in_coverage(self):
        rows = [{**pair("a", "test", 1), "match_score": .9, "accepted": True, "valid_aux": True},
                {**pair("b", "test", 0, "offscreen"), "match_score": 0., "accepted": False, "valid_aux": False},
                {**pair("c", "test", None, "noise"), "match_score": .7, "accepted": True, "valid_aux": True}]
        result = evaluate_records(rows, bootstrap=0)
        self.assertEqual(result["total_pairs"], 3); self.assertEqual(result["known_pairs"], 2)
        self.assertAlmostEqual(result["quality_valid_fraction"], 2/3)


class FixedGate(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", torch.tensor(logits, dtype=torch.float32))
        self.spec = {"audio_dim": 8}
    def forward(self, visual, audio, *args):
        return self.logits, torch.isfinite(audio).all((1, 2)), None


class RoutingPolicyTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        spec = dict(visual_dim=12, num_queries=4, hidden_size=8, num_heads=2, audio_dim=8,
                    architecture="phase4", value_mode="native", latent_norm="none")
        self.audio_router = AudioConditionedBottleneck(**spec, stage=3).eval().requires_grad_(False)
        self.vision_router = AudioConditionedBottleneck(**spec, stage=1).eval().requires_grad_(False)
        self.visual, self.audio, self.times = torch.randn(3, 9, 12), torch.randn(3, 5, 8), torch.tensor([0., 2., 7.])

    def test_disabled_policy_is_exact_original_path(self):
        expected = self.audio_router(self.visual, self.audio, self.times)
        policy = RelevanceRoutingPolicy(self.audio_router)
        actual = policy(self.visual, self.audio, self.times)
        for a, b in zip(actual, expected): self.assertTrue(torch.equal(a, b))
        self.assertIsNone(policy.last_gate_output)

    def test_provisional_calibration_requires_explicit_development_opt_in(self):
        calibration = {**CAL, "label_protocol": "weak_pairing"}
        with self.assertRaisesRegex(ValueError, "provisional"):
            RelevanceRoutingPolicy(self.audio_router, self.vision_router, FixedGate([20.]*3), calibration, True)
        policy = RelevanceRoutingPolicy(self.audio_router, self.vision_router, FixedGate([20.]*3), calibration,
                                        enabled=True, allow_provisional_calibration=True)
        policy(self.visual, self.audio, self.times, valid_aux=torch.ones(3, dtype=torch.bool))
        self.assertEqual(policy.last_gate_output.diagnostics["label_protocol"], "weak_pairing")

    def test_all_accepted_uses_original_audio_values_and_weights(self):
        expected = self.audio_router(self.visual, self.audio, self.times)
        policy = RelevanceRoutingPolicy(self.audio_router, self.vision_router, FixedGate([20.] * 3), CAL, True)
        actual = policy(self.visual, self.audio, self.times, valid_aux=torch.ones(3, dtype=torch.bool))
        for a, b in zip(actual, expected): self.assertTrue(torch.equal(a, b))
        self.assertTrue(all(not p.requires_grad for p in policy.parameters()))

    def test_all_rejected_is_exact_verified_vision_strategy(self):
        expected = self.vision_router(self.visual, None, self.times)
        policy = RelevanceRoutingPolicy(self.audio_router, self.vision_router, FixedGate([-20.] * 3), CAL, True)
        actual = policy(self.visual, self.audio, self.times, valid_aux=torch.ones(3, dtype=torch.bool))
        for a, b in zip(actual, expected): self.assertTrue(torch.equal(a, b))

    def test_mixed_batch_preserves_full_timestamp_deltas_no_interpolation(self):
        az, ap = self.audio_router(self.visual, self.audio, self.times)
        vz, vp = self.vision_router(self.visual, None, self.times)
        policy = RelevanceRoutingPolicy(self.audio_router, self.vision_router, FixedGate([20., -20., 20.]), CAL, True)
        z, p = policy(self.visual, self.audio, self.times, valid_aux=torch.ones(3, dtype=torch.bool))
        self.assertTrue(torch.equal(z[0], az[0])); self.assertTrue(torch.equal(z[2], az[2]))
        self.assertTrue(torch.equal(z[1], vz[1])); self.assertTrue(torch.equal(p[1], vp[1]))

    def test_missing_audio_uses_fallback(self):
        expected = self.vision_router(self.visual, None, self.times)
        policy = RelevanceRoutingPolicy(self.audio_router, self.vision_router, FixedGate([20.] * 3), CAL, True)
        actual = policy(self.visual, None, self.times, valid_aux=torch.zeros(3, dtype=torch.bool))
        for a, b in zip(actual, expected): self.assertTrue(torch.equal(a, b))

    def test_imagebind_policy_missing_audio_also_uses_fallback(self):
        expected = self.vision_router(self.visual, None, self.times)
        policy = RelevanceRoutingPolicy(self.audio_router, self.vision_router, ImageBindCosineGate(), CAL, True)
        actual = policy(self.visual, None, self.times, valid_aux=torch.zeros(3, dtype=torch.bool))
        for a, b in zip(actual, expected): self.assertTrue(torch.equal(a, b))
        self.assertFalse(policy.last_gate_output.accepted.any())

    def test_invalid_router_strategy_is_not_silently_used(self):
        with self.assertRaises(ValueError):
            RelevanceRoutingPolicy(self.audio_router, self.audio_router, FixedGate([20.] * 3), CAL, True)


if __name__ == "__main__":
    unittest.main()
