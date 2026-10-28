"""Utility regression, frozen selection policies and video-grouped statistics."""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import spearmanr


def safe_ratio(numerator, denominator):
    return float(numerator/denominator) if denominator else None


def arrays(rows):
    return {name: np.asarray([r[name] for r in rows]) for name in
            ('accuracy_utility','nll_utility','audio_correct','vision_correct','audio_gold_nll','vision_gold_nll')}


def selection_metrics(rows, accepted):
    values = arrays(rows)
    accepted = np.asarray(accepted, dtype=bool)
    if accepted.shape != (len(rows),):
        raise ValueError('one whole-prefix choice per QA required')
    correctness = np.where(accepted, values['audio_correct'], values['vision_correct'])
    nll = np.where(accepted, values['audio_gold_nll'], values['vision_gold_nll'])
    harmful = values['accuracy_utility'] == -1
    helpful = values['accuracy_utility'] == 1
    return {'n': len(rows), 'accuracy': float(correctness.mean()), 'gold_nll': float(nll.mean()),
        'gate_on_rate': float(accepted.mean()), 'audio_enabled_count': int(accepted.sum()),
        'false_allow_harmful': safe_ratio((accepted & harmful).sum(), harmful.sum()),
        'false_reject_helpful': safe_ratio((~accepted & helpful).sum(), helpful.sum()),
        'helpful_accepted': int((accepted & helpful).sum()), 'harmful_rejected': int((~accepted & harmful).sum()),
        'confusion': {'helpful_accepted': int((accepted & helpful).sum()), 'helpful_rejected': int((~accepted & helpful).sum()),
                      'harmful_accepted': int((accepted & harmful).sum()), 'harmful_rejected': int((~accepted & harmful).sum())},
        'accuracy_gain_over_audio': float(correctness.mean()-values['audio_correct'].mean()),
        'accuracy_gain_over_vision': float(correctness.mean()-values['vision_correct'].mean())}


def fit_threshold(rows, scores, valid):
    if not rows or any(r['split'] != 'val' for r in rows):
        raise ValueError('threshold calibration requires validation rows only')
    scores, valid = np.asarray(scores, dtype=float), np.asarray(valid, dtype=bool)
    if not np.isfinite(scores).all() or scores.shape != (len(rows),) or valid.shape != scores.shape:
        raise ValueError('nonfinite or malformed validation predictions')
    candidates = np.unique(np.concatenate(([float(scores.min())-max(1.,float(np.abs(scores).max()))], scores)))
    curve = []
    for threshold in candidates:
        metrics = selection_metrics(rows, (scores > threshold) & valid)
        curve.append({'threshold': float(threshold), **metrics})
    # Reproducible tie breaks prefer lower NLL, then less audio when both are identical.
    best = min(curve, key=lambda x: (-x['accuracy'], x['gold_nll'], x['gate_on_rate'], -x['threshold']))
    return {'threshold': best['threshold'], 'comparison': 'strict_greater_than', 'fit_split': 'val',
            'selection_objective': 'QA accuracy, then gold NLL, then lower audio rate',
            'validation': best, 'curve': curve}


def regression_metrics(rows, scores, accepted):
    from relevance_gate.evaluation import classification_metrics
    scores = np.asarray(scores, dtype=float)
    target = np.asarray([r['nll_utility'] for r in rows])
    error = scores-target
    result = {'MAE': float(np.abs(error).mean()),
        'Huber': float(np.where(np.abs(error) <= 1, .5*error**2, np.abs(error)-.5).mean()),
        'Spearman': None, **selection_metrics(rows, accepted)}
    if len(scores) > 2 and np.ptp(scores) > 0 and np.ptp(target) > 0:
        correlation = float(spearmanr(scores,target).statistic)
        result['Spearman'] = correlation if math.isfinite(correlation) else None
    mask = np.asarray([r['accuracy_utility'] != 0 for r in rows])
    if mask.any():
        binary = [int(r['accuracy_utility'] == 1) for r in rows if r['accuracy_utility'] != 0]
        result['discordant_QA_classification'] = classification_metrics(binary, scores[mask].tolist(), np.asarray(accepted)[mask].tolist(), probability=False)
    else:
        result['discordant_QA_classification'] = None
    # These bins are score-to-observed-utility reliability, not semantic probabilities.
    order = np.argsort(scores, kind='stable')
    result['utility_reliability_bins'] = [{'n': len(ix), 'predicted_utility': float(scores[ix].mean()),
        'observed_nll_utility': float(target[ix].mean()),
        'observed_accuracy_utility': float(np.mean([rows[i]['accuracy_utility'] for i in ix]))}
        for ix in np.array_split(order, min(5,len(order))) if len(ix)]
    return result


def random_selector(rows, valid, on_count, seed=1234):
    rng = np.random.default_rng(seed)
    eligible = np.flatnonzero(np.asarray(valid,dtype=bool))
    if on_count > len(eligible):
        raise ValueError('random selector cannot exceed eligible audio count')
    result = np.zeros(len(rows),dtype=bool)
    result[rng.choice(eligible, int(on_count), replace=False)] = True
    return result


def video_bootstrap_gain(rows, accepted, reference='audio', seed=1234, repetitions=1000):
    groups = sorted({r['source_video_id'] for r in rows})
    if len(groups) < 5:
        return {'ci95': None, 'reason': 'fewer than five independent source groups'}
    accepted = np.asarray(accepted,dtype=bool)
    source = np.asarray([r['source_video_id'] for r in rows])
    chosen = np.asarray([r['audio_correct'] if a else r['vision_correct'] for r,a in zip(rows,accepted)],float)
    ref = np.asarray([r[reference+'_correct'] for r in rows],float)
    numerator = np.asarray([(chosen[source == group]-ref[source == group]).sum() for group in groups])
    denominator = np.asarray([(source == group).sum() for group in groups])
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(groups),size=(repetitions,len(groups)))
    gains = numerator[draws].sum(1)/denominator[draws].sum(1)
    return {'ci95': np.quantile(gains,[.025,.975]).tolist(), 'groups': len(groups),
            'repetitions': repetitions, 'resampling_unit': 'source_video_id'}
