"""Frozen Gate2 scores against official LLP joint-event evaluation labels."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path

import numpy as np
import torch

from utility_gate.common import append_jsonl, read_jsonl, seed_all, write_json
from utility_gate.gate2_temporal import Gate2OnlineState
from utility_gate.strict_runtime import StrictStreamingRuntime
from .dataset_audit import ROOT
from .llp_parser import labels_at
from .native import encode_times
from .runtime import gates, lock_evaluation
from .stats import binary_cluster_ci, binary_metrics, mean_ci


def association(rows, target):
    sources = [r['source_video_id'] for r in rows]
    result = binary_cluster_ci([r[target] for r in rows], [r['gate_raw_utility'] for r in rows],
                               [r['gate_decision'] for r in rows], sources)
    by_source = defaultdict(list)
    for r in rows: by_source[r['source_video_id']].append(r)
    differences, selected = [], []
    for source, group in by_source.items():
        positive = [r['gate_raw_utility'] for r in group if r[target]]
        negative = [r['gate_raw_utility'] for r in group if not r[target]]
        if positive and negative:
            differences.append(float(np.mean(positive)-np.mean(negative))); selected.append(source)
    result['within_source_joint_minus_nonjoint_score'] = mean_ci(differences, selected)
    return result


def plot(output, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve
    if not rows: return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    labels = ['neither', 'visual_only', 'audio_only', 'different_categories', 'joint_same_class']
    data = [[r['gate_raw_utility'] for r in rows if r['subtype'] == name] for name in labels]
    names, values = zip(*[(k, v) for k, v in zip(labels, data) if v])
    axes[0].boxplot(values, tick_labels=names, showfliers=False)
    axes[0].axhline(rows[0]['gate_threshold'], color='red', linestyle='--', label='Frozen threshold')
    axes[0].tick_params(axis='x', labelrotation=30)
    axes[0].set(title='Available audio only', ylabel='Raw utility score'); axes[0].legend(fontsize=8)
    y = np.array([r['current_bin_joint'] for r in rows]); scores = np.array([r['gate_raw_utility'] for r in rows])
    if y.any() and (~y).any():
        fpr, tpr, _ = roc_curve(y, scores); precision, recall, _ = precision_recall_curve(y, scores)
        axes[1].plot(fpr, tpr); axes[1].plot([0, 1], [0, 1], '--', color='gray')
        axes[2].plot(recall, precision); axes[2].axhline(y.mean(), linestyle='--', color='gray')
    axes[1].set(xlabel='FPR', ylabel='TPR', title='Joint current-bin ROC')
    axes[2].set(xlabel='Recall', ylabel='Precision', title='Joint current-bin PR')
    fig.savefig(output/'temporal_scores.png', dpi=170); plt.close(fig)


def run(root=ROOT):
    root = Path(root); output = root/'temporal'; output.mkdir(exist_ok=True)
    lock_evaluation(root, 'llp')
    seed_all(1234); runtime = StrictStreamingRuntime()
    _, _, gate, threshold = gates(runtime.device)
    path = output/'per_second.jsonl'
    prior = read_jsonl(path) if path.exists() else []
    done = {r['clip_id'] for r in prior}
    manifests = read_jsonl(root/'datasets/llp_manifest.jsonl')
    with torch.inference_mode():
        for i, row in enumerate(manifests):
            if row['clip_id'] in done: continue
            try:
                times = [t for t in range(1, 11) if t <= row['duration_s']+1e-6]
                payload, _ = encode_times(runtime, row['media_path'], times)
                total_joint = sum(labels_at(row['audio_events'], row['visual_events'], t)['current_bin_joint'] for t in range(1, 11))
                for grid, indices in [('1s', list(range(len(times)))), ('2s', [j for j, t in enumerate(times) if t % 2 == 0])]:
                    state = Gate2OnlineState(runtime, gate, threshold)
                    for j in indices:
                        t = times[j]; labels = labels_at(row['audio_events'], row['visual_events'], t)
                        state.step(payload['visual'][j], payload['causal_audio'][j], t,
                            bool(payload['audio_valid'][j]), t, payload['frame_pts_s'][j])
                        trace = state.trace[-1]
                        result = {'clip_id': row['clip_id'], 'source_video_id': row['source_video_id'],
                            'media_sha256': row['media_sha256'], 'clip_offset_s': row['clip_offset_s'],
                            'grid': grid, 'time_s': t, 'frame_time': payload['frame_pts_s'][j],
                            'frame_lag_s': t-payload['frame_pts_s'][j], 'audio_start_s': max(0, t-2), 'audio_end_s': t,
                            'gate_raw_utility': trace['utility_score'], 'gate_threshold': threshold,
                            'gate_decision': trace['accepted'], 'sigmoid_diagnostic': trace['gate'],
                            'audio_available': trace['availability'], 'audio_quality_status': payload['audio_quality'][j],
                            'joint_seconds_in_full_annotation': total_joint, **labels,
                            'labels_passed_to_gate': False, 'historical_tokens_revised': False}
                        append_jsonl(path, result)
            except Exception as error:
                append_jsonl(output/'failures.jsonl', {'clip_id': row['clip_id'], 'reason': str(error)})
            if (i+1) % 8 == 0: print({'temporal_clips_attempted': i+1}, flush=True)
    records = read_jsonl(path) if path.exists() else []
    summary = {'manifest_videos': len(manifests), 'measured_videos': len({r['clip_id'] for r in records}),
        'frozen_threshold': threshold, 'grids': {}, 'new_threshold_fits': 0,
        'scope': 'joint-event association, NOT a true downstream utility target',
        '1s_grid_distribution_shift': 'same measured-window model trained on2s grid, no adaptation',
        '2s_grid': 'separate chronological state at2/4/6/8/10, not filtered1s history'}
    for grid in ('1s', '2s'):
        rows = [r for r in records if r['grid'] == grid]
        subsets = {'all_scheduled_with_availability_mask': rows,
            'quality_qualified': [r for r in rows if r['audio_available']],
            'PTS_ok_only': [r for r in rows if r['audio_quality_status'] == 'ok']}
        value = {'scheduled_bins': len(rows), 'quality_counts': dict(Counter(r['audio_quality_status'] for r in rows)),
            'subtypes_all': dict(Counter(r['subtype'] for r in rows)), 'denominators': {}}
        for name, subset in subsets.items():
            value['denominators'][name] = {target: association(subset, target) for target in ('current_bin_joint', 'causal_window_joint')}
        value['quality_qualified_equals_PTS_ok'] = len(subsets['quality_qualified']) == len(subsets['PTS_ok_only'])
        by_type = defaultdict(list); by_category = defaultdict(list); by_duration = defaultdict(list)
        for r in subsets['quality_qualified']:
            by_type[r['subtype']].append(r)
            for category in set(r['audio_categories']+r['visual_categories']): by_category[category].append(r)
            seconds = r['joint_seconds_in_full_annotation']
            by_duration['0' if seconds == 0 else '1-3' if seconds <= 3 else '4-7' if seconds <= 7 else '8-10'].append(r)
        basic = lambda v: {**binary_metrics([r['current_bin_joint'] for r in v], [r['gate_raw_utility'] for r in v], [r['gate_decision'] for r in v]),
                           'sources': len({r['source_video_id'] for r in v}), 'mean_score': float(np.mean([r['gate_raw_utility'] for r in v]))}
        value['available_by_subtype'] = {k: basic(v) for k, v in by_type.items()}
        value['available_by_event_category'] = {k: basic(v) for k, v in by_category.items()}
        value['available_by_joint_duration_seconds'] = {k: basic(v) for k, v in by_duration.items()}
        multi = [r for r in subsets['quality_qualified'] if len(set(r['audio_categories']+r['visual_categories'])) > 1]
        value['available_multi_event'] = basic(multi) if multi else {'n': 0}
        summary['grids'][grid] = value
    write_json(output/'summary.json', summary)
    if records:
        with (output/'per_second.csv').open('w') as stream:
            fields = ['clip_id', 'source_video_id', 'grid', 'time_s', 'frame_time', 'audio_start_s', 'audio_end_s',
                'gate_raw_utility', 'gate_threshold', 'gate_decision', 'sigmoid_diagnostic', 'audio_available',
                'audio_quality_status', 'current_bin_joint', 'causal_window_joint', 'subtype']
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(records)
    plot(output, [r for r in records if r['grid'] == '1s' and r['audio_available']])
    print({'videos': summary['measured_videos'], 'bins': len(records), 'threshold_fits': 0})


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, default=ROOT)
    run(p.parse_args().output)


if __name__ == '__main__': main()
