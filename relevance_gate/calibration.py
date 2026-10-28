"""Validation-only monotone Platt calibration and tie-consistent thresholds."""

import hashlib
import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from .protocols import TRUSTED, WEAK


def calibrate(records, model_fingerprint, target_tpr=0.95):
    if not records or any(row["split"] != "val" for row in records):
        raise ValueError("calibration can only use the validation split")
    protocols = {row.get("label_protocol", TRUSTED) for row in records}
    if len(protocols) != 1 or not protocols.issubset({TRUSTED, WEAK}):
        raise ValueError("calibration cannot mix semantic and weak-pairing targets")
    protocol = next(iter(protocols))
    labeled = [row for row in records if row["visible_correspondence"] is not None]
    usable = [row for row in labeled if row["valid_aux"]]
    labels = np.array([row["visible_correspondence"] for row in usable])
    values = np.array([row["raw_score"] for row in usable], dtype=np.float64)
    if len(np.unique(labels)) != 2 or not np.isfinite(values).all():
        raise ValueError("calibration requires finite, valid validation positives AND negatives")
    if not 0 < target_tpr <= 1:
        raise ValueError("target_tpr must be in (0,1]")

    def loss(parameters):
        logits = values * parameters[0] + parameters[1]
        return np.mean(np.logaddexp(0., logits) - labels * logits) + 1e-6 * parameters[0] ** 2

    fit = minimize(loss, [1., 0.], method="L-BFGS-B", bounds=[(0., None), (None, None)])
    if not fit.success:
        raise RuntimeError(f"validation calibration failed: {fit.message}")
    scale, bias = map(float, fit.x)
    all_labels = np.array([row["visible_correspondence"] for row in labeled])
    valid = np.array([row["valid_aux"] for row in labeled], dtype=bool)
    probabilities = expit(np.array([row["raw_score"] for row in labeled]) * scale + bias)
    probabilities[~valid] = 0.
    positives = all_labels == 1
    max_tpr = float(valid[positives].mean())
    effective_target = min(target_tpr, max_tpr)
    candidates = sorted(set(probabilities[valid]), reverse=True)
    threshold = 0.
    reached = False
    for value in candidates:
        if ((probabilities >= value) & valid & positives).sum() / positives.sum() + 1e-12 >= effective_target:
            threshold, reached = float(value), True
            break
    payload = {"version": 2, "method": "validation_monotone_platt", "scale": scale, "bias": bias,
               "label_protocol": protocol, "semantic_validation_complete": False,
               "threshold": threshold, "comparison": ">=", "target_tpr": target_tpr,
               "target_attainable": reached and max_tpr + 1e-12 >= target_tpr, "max_quality_limited_tpr": max_tpr,
               "effective_validation_tpr_target": effective_target,
               "threshold_policy": "highest validation threshold attaining requested or quality-limited TPR; no all-valid threshold-zero fallback",
               "validation_pairs": len(labeled), "fit_pairs": len(usable),
               "validation_source_videos": sorted({v for r in records for v in (r['source_video_id'], r.get('audio_source_video_id')) if v}),
               "model_fingerprint": model_fingerprint,
               "validation_sha256": hashlib.sha256(json.dumps(records, sort_keys=True, allow_nan=False).encode()).hexdigest()}
    return payload


def probabilities(raw, valid, calibration):
    values = expit(np.asarray(raw) * calibration["scale"] + calibration["bias"])
    return np.where(valid, values, 0.)
