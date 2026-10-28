"""Honest source-level intervals; no IID inflation from repeated windows."""
import numpy as np

from utility_gate.grounding.stats import mean_ci


def paired_source_interval(values,sources):
    if len(set(sources))<2:
        return {'status':'insufficient_data','n':len(values),'sources':len(set(sources)),
                'ci95':None,'reason':'fewer than two independent source clusters'}
    return {'status':'descriptive_source_bootstrap',**mean_ci(values,sources)}


def spearman_source_interval(predicted, target, sources, repetitions=2000, seed=1234):
    from scipy.stats import spearmanr
    predicted,target=np.asarray(predicted,float),np.asarray(target,float)
    if len(predicted)!=len(target) or len(target)!=len(sources):
        raise ValueError('paired scores, targets and sources required')
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError('finite regression pairs required')
    groups=sorted(set(sources))
    if len(groups)<2 or np.ptp(predicted)==0 or np.ptp(target)==0:
        return {'status':'insufficient_data','coefficient':None,'ci95':None,
                'reason':'constant scores/targets or fewer than two source clusters'}
    indices=[np.flatnonzero(np.asarray(sources)==group) for group in groups]
    values=[];rng=np.random.default_rng(seed)
    for _ in range(repetitions):
        selected=np.concatenate([indices[i] for i in rng.integers(0,len(groups),len(groups))])
        if np.ptp(predicted[selected]) and np.ptp(target[selected]):
            values.append(float(spearmanr(predicted[selected],target[selected]).statistic))
    return {'status':'descriptive_source_bootstrap','coefficient':float(spearmanr(predicted,target).statistic),
            'ci95':np.quantile(values,[.025,.975]).tolist() if values else None,
            'valid_resamples':len(values),'sources':len(groups),'n':len(target),
            'seed':seed,'repetitions':repetitions,'unit':'original source video',
            'small_n_warning':len(groups)<20}
