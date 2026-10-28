"""Global/stratified correspondence metrics; quality and semantic labels separate."""

from collections import Counter

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from .manifest import STRATA


def classification_metrics(labels, scores, accepted, probability=True):
    labels, scores, accepted = np.asarray(labels), np.asarray(scores), np.asarray(accepted, dtype=bool)
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite evaluation scores")
    positives = labels == 1
    negatives = labels == 0
    two_classes = positives.any() and negatives.any()
    fpr95 = None
    if two_classes and positives.sum() >= 20 and negatives.sum() >= 20:
        fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
        fpr95 = float(fpr[tpr >= 0.95].min())
    result = {"examples": len(labels), "positives": int(positives.sum()), "negatives": int(negatives.sum()),
              "auroc": float(roc_auc_score(labels, scores)) if two_classes else None,
              "auprc": float(average_precision_score(labels, scores)) if positives.any() else None,
              "fpr_at_tpr95": fpr95,
              "fpr95_support": "requires at least 20 positives and 20 negatives; empirical ROC, not deployment TPR",
              "frozen_threshold_tpr": float(accepted[positives].mean()) if positives.any() else None,
              "false_accept_rate": float(accepted[negatives].mean()) if negatives.any() else None,
              "brier": None, "ece_10_bins": None}
    if probability and len(labels):
        if (scores < 0).any() or (scores > 1).any():
            raise ValueError("Brier/ECE require probabilities, never raw cosine")
        result["brier"] = float(((scores - labels) ** 2).mean())
        bin_ids = np.minimum((scores * 10).astype(int), 9)
        result["ece_10_bins"] = float(sum(
            (bin_ids == index).mean() * abs(scores[bin_ids == index].mean() - labels[bin_ids == index].mean())
            for index in range(10) if (bin_ids == index).any()))
    return result


def evaluate_records(records, bootstrap=200, seed=1234):
    if not records:
        raise ValueError("no predictions to evaluate")
    known = [row for row in records if row["visible_correspondence"] is not None]

    def measure(rows):
        return classification_metrics([r["visible_correspondence"] for r in rows],
                                      [r["match_score"] for r in rows], [r["accepted"] for r in rows])

    positives = [row for row in known if row["visible_correspondence"] == 1]
    groups = {}
    for stratum in STRATA:
        rows = [row for row in records if row["negative_type"] == stratum]
        labeled = [r for r in rows if r["visible_correspondence"] is not None]
        negatives = [r for r in labeled if r["visible_correspondence"] == 0]
        groups[stratum] = {"total": len(rows), "known": len(labeled), "unknown": len(rows) - len(labeled),
                           "valid_aux_fraction": sum(r["valid_aux"] for r in rows) / len(rows) if rows else None,
                           "accept_fraction": sum(r["accepted"] for r in rows) / len(rows) if rows else None,
                           "metrics": measure(labeled),
                           "versus_all_positives": measure(positives + negatives) if negatives else None}
    result = {"total_pairs": len(records), "known_pairs": len(known), "unknown_pairs": len(records) - len(known),
              "quality_valid_fraction": sum(r["valid_aux"] for r in records) / len(records),
              "quality_reasons": dict(Counter(r.get("quality_reason", "unspecified") for r in records)),
              "overall": measure(known), "valid_features_only": measure([r for r in known if r["valid_aux"]]),
              "by_stratum": groups, "source_video_bootstrap_ci95": {}}
    parents = {}
    def find(value):
        parents.setdefault(value, value)
        root = value
        while parents[root] != root:
            root = parents[root]
        while parents[value] != value:
            parent = parents[value]
            parents[value] = root
            value = parent
        return root
    for row in known:
        visual_id = row["source_video_id"]
        audio_id = row.get("audio_source_video_id") or visual_id
        parents[find(audio_id)] = find(visual_id)
    ids = sorted({find(r["source_video_id"]) for r in known})
    result["independent_source_components"] = len(ids)
    if len(ids) >= 5 and bootstrap:
        rng = np.random.default_rng(seed)
        estimates = []
        by_id = {video: [r for r in known if find(r["source_video_id"]) == video] for video in ids}
        for _ in range(bootstrap):
            sample = [row for video in rng.choice(ids, len(ids), replace=True) for row in by_id[video]]
            if len({r["visible_correspondence"] for r in sample}) == 2:
                estimates.append(measure(sample))
        for key in ("auroc", "auprc", "false_accept_rate", "brier", "ece_10_bins"):
            values = [row[key] for row in estimates if row[key] is not None]
            if len(values) >= 20:
                result["source_video_bootstrap_ci95"][key] = np.quantile(values, [.025, .975]).tolist()
    result["bootstrap_dependency_note"] = "clusters are connected visual/audio source components; fewer than 5 independent components yields no CI"
    return result


def failure_examples(records, limit=100):
    errors = [r for r in records if r["visible_correspondence"] is not None and bool(r["accepted"]) != bool(r["visible_correspondence"])]
    return sorted(errors, key=lambda r: abs(r["match_score"] - r["visible_correspondence"]), reverse=True)[:limit]
