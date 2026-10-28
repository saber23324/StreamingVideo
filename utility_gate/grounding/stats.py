"""Source-cluster intervals for observed grounding and paired QA outcomes."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def mean_ci(values, sources, repetitions=2000, seed=1234):
    values = np.asarray(values, float)
    if not len(values):
        return {'n': 0, 'sources': 0, 'mean': None, 'ci95': None}
    if not np.isfinite(values).all() or len(values) != len(sources):
        raise ValueError('finite values and source identities required')
    groups = sorted(set(sources))
    sums = np.asarray([values[np.asarray(sources) == g].sum() for g in groups])
    counts = np.asarray([sum(x == g for x in sources) for g in groups])
    index = np.random.default_rng(seed).integers(0, len(groups), (repetitions, len(groups)))
    samples = sums[index].sum(1)/counts[index].sum(1)
    return {'n': len(values), 'sources': len(groups), 'mean': float(values.mean()),
            'ci95': np.quantile(samples, [.025, .975]).tolist(), 'repetitions': repetitions,
            'unit': 'original source video', 'small_n_warning': len(groups) < 20}


def binary_metrics(labels, scores, decisions):
    y, s, d = np.asarray(labels, bool), np.asarray(scores, float), np.asarray(decisions, bool)
    p, n = int(y.sum()), int((~y).sum())
    tpr = float(d[y].mean()) if p else None
    fpr = float(d[~y].mean()) if n else None
    enough = p >= 5 and n >= 5
    return {'n': len(y), 'positives': p, 'negatives': n,
            'enable_rate': float(d.mean()) if len(d) else None,
            'TPR': tpr, 'FPR': fpr, 'balanced_accuracy': (tpr+1-fpr)/2 if p and n else None,
            'AUROC': float(roc_auc_score(y, s)) if enough else None,
            'AUPRC': float(average_precision_score(y, s)) if enough else None,
            'score_interpretation': 'raw frozen utility regression, not a calibrated probability',
            'classification_support': 'at least5 bins each required; bins within a video are dependent'}


def binary_cluster_ci(labels, scores, decisions, sources, repetitions=1000, seed=1234):
    result = binary_metrics(labels, scores, decisions)
    groups = sorted(set(sources))
    result['sources'] = len(groups)
    result['cluster_ci95'] = {}
    if not groups:
        return result
    y, s, d = np.asarray(labels), np.asarray(scores), np.asarray(decisions)
    indices = [np.flatnonzero(np.asarray(sources) == g) for g in groups]
    rng = np.random.default_rng(seed)
    samples = {k: [] for k in ('enable_rate', 'TPR', 'FPR', 'balanced_accuracy', 'AUROC', 'AUPRC')}
    for _ in range(repetitions):
        selected = np.concatenate([indices[i] for i in rng.integers(0, len(groups), len(groups))])
        value = binary_metrics(y[selected], s[selected], d[selected])
        for key in samples:
            if value[key] is not None:
                samples[key].append(value[key])
    for key, values in samples.items():
        result['cluster_ci95'][key] = {'ci95': np.quantile(values, [.025, .975]).tolist() if values else None,
                                       'valid_resamples': len(values)}
    result.update(repetitions=repetitions, small_n_warning=len(groups) < 20)
    return result
