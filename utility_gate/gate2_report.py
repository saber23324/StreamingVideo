"""Aggregate completed Gate2 runs without fitting or executing any policy."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

from .common import (AUDIO_ROUTER, DATA_ROOT, REPO, VISION_ROUTER, read_jsonl,
                     sha256, validate_splits, write_json, write_jsonl)


def load(path):
    return json.loads(Path(path).read_text())


def mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def paired_interval(rows, first, second, seed=1234):
    """One selected QA per original source; paired source bootstrap."""
    if len({r['source_video_id'] for r in rows}) != len(rows):
        raise ValueError('this bootstrap requires exactly one QA per source')
    delta = np.asarray([int(r[first])-int(r[second]) for r in rows])
    generator = np.random.default_rng(seed)
    samples = delta[generator.integers(0, len(rows), (10000, len(rows)))].mean(1)
    return {'gain': float(delta.mean()), 'ci95': np.quantile(samples, [.025, .975]).tolist(),
            'resampling_unit': 'original YouTube source', 'repetitions': 10000,
            'interpretation': 'conditional on this selected subset; no benchmark-wide claim'}


def audit(root, data):
    manifest = read_jsonl(data/'qa_manifest.jsonl')
    source_audit = validate_splits(manifest)
    assert len(manifest) == 896
    assets = {}
    for directory in ('qualified_labels', 'temporal_labels'):
        for split in ('train', 'val'):
            path = root/directory/'training_assets'/(split+'.jsonl')
            rows = read_jsonl(path)
            assert all(r['split'] == split and r['official_split'] == 'train' for r in rows)
            assert not any(r['question_id'] in {m['question_id'] for m in manifest if m['split'] == 'test'} for r in rows)
            assets[str(path.relative_to(root))] = {'rows': len(rows), 'sha256': sha256(path)}
    locked = load(root/'primary_model_locked_before_test.json')
    assert sha256(root/'qualified_model/selection.json') == locked['model_selection_sha256']
    assert sha256(root/'qualified_model/train_config.json') == locked['train_config_sha256']
    time_lock = load(root/'temporal_locked_before_test.json')
    assert sha256(root/'temporal_calibration/streaming_threshold.json') == time_lock['calibration_sha256']
    assert sha256(load(root/'temporal_calibration/streaming_threshold.json')['model_checkpoint']) == time_lock['model_sha256']
    m0 = load(root/'m0/integrity.json')
    assert sha256(AUDIO_ROUTER) == m0['audio_checkpoint_sha256']
    assert sha256(VISION_ROUTER) == m0['vision_checkpoint_sha256']
    baseline_sources = m0['runtime_metadata']['source_sha256']
    assert all(sha256(REPO/path) == digest for path, digest in baseline_sources.items())
    source_at_fit = root/'qualified_source_at_fit/gate2_experiment.py'
    for model in ('qualified_model', 'temporal_model'):
        config = load(root/model/'train_config.json')
        fit_source = source_at_fit if model == 'qualified_model' else REPO/'utility_gate/gate2_experiment.py'
        assert config['experiment_source_sha256'] == sha256(fit_source)
        assert config['feature_source_sha256'] == sha256(REPO/'utility_gate/gate2_interfaces.py')
        assert not config['test_targets_read_during_fit']
    # Store the final sources in addition to the hash-verified training source.
    paths = sorted((REPO/'utility_gate').glob('*.py')) + [REPO/'tests/test_gate2.py']
    snapshots = root/'final_source'
    hashes = {}
    for path in paths:
        relative = path.relative_to(REPO)
        target = snapshots/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[str(relative)] = sha256(path)
    media = [r for r in manifest if sha256(r['video_path']) != r['media_sha256']]
    assert not media
    tests = (data/'logs/full_tests.log').read_text()
    match = re.search(r'Ran (\d+) tests in ([\d.]+)s\s+OK\s*$', tests)
    assert match is not None, 'full test log must end with an actual success'
    versions = {name: importlib.metadata.version(name) for name in
                ('torch', 'transformers', 'numpy', 'scikit-learn', 'av', 'matplotlib')}
    result = {'source_groups': source_audit, 'all_896_media_hashes_verified': True,
              'training_assets': assets, 'test_targets_in_training_assets': False,
              'pretest_locks_still_match': True, 'owner_checkpoints_unchanged': True,
              'protected_baseline_sources_unchanged': True, 'fit_sources_hash_verified': True,
              'source_at_fit_note': 'hash-exact copy reconstructed by reverting only the reporting row name legacy_always_audio -> always_audio_endpoint',
              'final_source_sha256': hashes, 'versions': versions,
              'tests': {'passed': int(match[1]), 'seconds': float(match[2]), 'log': str(data/'logs/full_tests.log')}}
    write_json(root/'final_audit.json', result)
    return result


def plot_summary(root, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    groups = [
        ('AVQA held-out prefix proxy (n=256)', summary['prefix']['methods'],
         ['full_vision', 'vision64', 'legacy_audio64', 'qualified_audio64', 'learned_gate', 'random_matched_rate']),
        ('Chronological decisions (n=64)', summary['temporal']['methods'],
         ['full_vision', 'vision64', 'legacy_audio64', 'qualified_audio64', 'learned_gate', 'causal_random'])]
    labels = ['Full vision', 'Vision 64', 'Legacy audio 64', 'Valid audio 64', 'Learned gate', 'Random']
    for ax, (title, values, order) in zip(axes[0], groups):
        scores = [100*values[k]['accuracy'] for k in order]
        bars = ax.barh(labels[::-1], scores[::-1], color=['#8274b3', '#2a9d8f', '#d5a34b', '#d27756', '#577d9e', '#a1a8ad'][::-1])
        ax.set(xlim=(0, 106), xlabel='Accuracy (%)', title=title)
        for bar, value in zip(bars, scores[::-1]):
            ax.text(value+1, bar.get_y()+bar.get_height()/2, f'{value:.2f}', va='center', fontsize=9)
    ax = axes[1, 0]
    conditions = summary['robustness']['conditions']
    keys = list(conditions)
    names = ['Missing', 'Silence', 'Noise 0 dB', 'Past shift 2s', 'Wrong audio']
    x = np.arange(len(keys))
    for offset, field, name, color in [(-.24, 'vision_accuracy', 'Vision 64', '#577d9e'),
                                      (0, 'safe_audio_accuracy', 'Valid audio 64', '#d5a34b'),
                                      (.24, 'gated_accuracy', 'Learned gate', '#2a9d8f')]:
        ax.bar(x+offset, [100*conditions[k][field] for k in keys], .24, label=name, color=color)
    ax.set(xticks=x, xticklabels=names, ylim=(0, 105), ylabel='Accuracy (%)', title='Frozen gate under controlled inputs (64 per condition)')
    ax.tick_params(axis='x', labelrotation=15)
    ax.legend(loc='lower left', fontsize=9)
    ax = axes[1, 1]
    timings = summary['latency']['methods']
    keys = ['full_vision', 'always_off', 'legacy_audio', 'safe_audio', 'energy', 'learned']
    names = ['Full vision', 'Vision 64', 'Legacy audio', 'Valid audio', 'Energy gate', 'Learned gate']
    y = np.arange(len(keys))
    ax.barh(y, [timings[k]['pipeline_total_s'] for k in keys], color='#577d9e', label='End to end')
    ax.scatter([timings[k]['pipeline_ttft_s'] for k in keys], y, color='#d27756', s=24, label='TTFT', zorder=3)
    for i, k in enumerate(keys):
        ax.text(timings[k]['pipeline_total_s']+.015, i, f"{timings[k]['pipeline_total_s']:.3f}s", va='center', fontsize=9)
    ax.set(yticks=y, yticklabels=names, xlim=(0, 1.2), xlabel='Median seconds', title='Real media + encoders + VLM (n=8, warm shared H800)')
    ax.invert_yaxis()
    ax.legend(loc='lower left', fontsize=9)
    fig.suptitle('Gate2 observed results: no improvement over the fixed vision branch', fontsize=14)
    fig.savefig(root/'observed_results.png', dpi=180)
    fig.savefig(root/'observed_results.svg')
    plt.close(fig)


def aggregate(root, data):
    root, data = Path(root), Path(data)
    paired = read_jsonl(root/'qualified_paired/paired_qa.jsonl')
    test = [r for r in paired if r['split'] == 'test']
    actual = read_jsonl(root/'integration/gated_results.jsonl')
    assert len(test) == len(actual) == 256
    policies = {r['question_id']: r for r in actual}
    assert set(policies) == {r['question_id'] for r in test}
    for row in test:
        selected = 'audio' if policies[row['question_id']]['accepted_prefix'] else 'vision'
        assert policies[row['question_id']]['generated_ids'] == row['branches'][selected]['generated_ids']
        assert all(row['frame_pts_s'][i] <= t+1e-5 for i, t in enumerate(row['timestamps_s']))
        assert all(r['visual_tokens'] == 64*len(row['timestamps_s']) for k, r in row['branches'].items() if k != 'full_vision')
    metrics = load(root/'qualified_model/test_metrics.json')
    primary = next(r for r in metrics if r['method'] == 'anchor_history' and r['seed'] == 1234)
    methods = {name: {'accuracy': float(np.mean([r['branches'][branch]['correct'] for r in test])),
                       'correct': sum(r['branches'][branch]['correct'] for r in test), 'n': len(test)}
               for name, branch in [('full_vision', 'full_vision'), ('vision64', 'vision'),
                                    ('legacy_audio64', 'legacy_audio'), ('qualified_audio64', 'audio')]}
    methods['learned_gate'] = primary
    for name in ('random_matched_rate', 'energy_gate', 'offline_endpoint_oracle'):
        methods[name] = next(r for r in metrics if r['method'] == name)
    per_modality = {}
    failures = []
    for modality in ('View', 'Sound', 'Both'):
        rows = [r for r in test if r['question_modality'] == modality]
        per_modality[modality] = {'n': len(rows), **{k: float(np.mean([r['branches'][b]['correct'] for r in rows]))
            for k, b in [('vision', 'vision'), ('raw_audio', 'legacy_audio'), ('qualified_audio', 'audio'), ('full_vision', 'full_vision')]},
            'gate': float(np.mean([policies[r['question_id']]['correct'] for r in rows])),
            'prefix_accepted': sum(policies[r['question_id']]['accepted_prefix'] for r in rows)}
    for row in test:
        a, v, g = row['branches']['audio']['correct'], row['branches']['vision']['correct'], policies[row['question_id']]
        kind = 'missed_helpful_audio' if a and not v and not g['correct'] else (
            'harmful_audio_avoided' if v and not a and g['correct'] else None)
        if kind:
            failures.append({'scope': 'prefix', 'kind': kind, 'question_id': row['question_id'],
                             'source_video_id': row['source_video_id'], 'question_modality': row['question_modality'],
                             'vision': row['branches']['vision'], 'qualified_audio': row['branches']['audio'], 'gate': g})
    temporal = load(root/'temporal_evaluation/summary.json')
    temporal_rows = read_jsonl(root/'temporal_evaluation/streaming_results.jsonl')
    temporal_methods = {name: {'accuracy': temporal[field], 'n': temporal['n']} for name, field in
                        [('full_vision', 'full_vision_accuracy'), ('vision64', 'vision_accuracy'),
                         ('legacy_audio64', 'legacy_audio_accuracy'), ('qualified_audio64', 'safe_audio_accuracy'),
                         ('learned_gate', 'accuracy'), ('causal_random', 'random_accuracy')]}
    for row in temporal_rows:
        for event in row['trace']:
            assert event['source_audio_end_s'] <= event['timestamp_s']+1e-6
            assert not event['historical_tokens_revised']
        if row['vision_correct'] and not row['correct']:
            failures.append({'scope': 'chronological', 'kind': 'gate_harms_correct_vision', **row})
    robustness = load(root/'robustness/summary.json')
    for row in read_jsonl(root/'robustness/robustness_results.jsonl'):
        assert row['source_audio_ends_causal']
        if row['vision_correct'] and not row['gated_correct']:
            failures.append({'scope': 'robustness', 'kind': 'gate_harms_correct_vision', **row})
    qa = {r['question_id']: r for r in read_jsonl(data/'qa_manifest.jsonl')}
    for row in failures:
        row.update(question=qa[row['question_id']]['question'], options=qa[row['question_id']]['options'],
                   gold_answer=qa[row['question_id']]['answer'], video_path=qa[row['question_id']]['video_path'])
    write_jsonl(root/'failure_examples.jsonl', failures)
    quality = Counter(q for r in paired for q in r['audio_quality'])
    stamps = Counter(len(r['timestamps_s']) for r in paired)
    result = {'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'active_instruction': 'AGENTS_gate2.md', 'data': load(data/'split_manifest.json'),
              'audio_quality_windows': dict(quality), 'frames_per_clip': dict(stamps),
              'prefix': {'methods': methods, 'per_official_modality': per_modality,
                         'replicates': [r for r in metrics if r['method'] in ('anchor_history', 'pooled_history')],
                         'actual_generations': load(root/'integration/summary.json')},
              'temporal': {'methods': temporal_methods, 'details': temporal,
                           'paired_gain_over_vision': paired_interval(temporal_rows, 'correct', 'vision_correct')},
              'raw_replay': load(root/'raw_replay/summary.json'), 'robustness': robustness,
              'null_audio_control': load(root/'null_audio_control/summary.json'),
              'latency': load(root/'latency/summary.json'), 'audit': audit(root, data),
              'failure_counts': dict(Counter(r['scope']+':'+r['kind'] for r in failures)),
              'conclusion': 'No demonstrated accuracy or end-to-end latency improvement over fixed vision64.',
              'not_established': ['semantic audiovisual correspondence', 'physical visual-evidence retention mechanism',
                                  'matched-training no-audio ablation', 'real sensor modalities', 'cross-VLM transfer']}
    comparison = [{'source_video_id': r['source_video_id'], 'gate': policies[r['question_id']]['correct'],
                   'vision': r['branches']['vision']['correct']} for r in test]
    result['prefix']['paired_gain_over_vision'] = paired_interval(comparison, 'gate', 'vision')
    write_json(root/'summary.json', result)
    with (root/'observed_results.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['experiment', 'method', 'n', 'accuracy'])
        writer.writeheader()
        for experiment, values in [('prefix', methods), ('chronological', temporal_methods)]:
            for method, value in values.items():
                writer.writerow({'experiment': experiment, 'method': method, 'n': value['n'], 'accuracy': value['accuracy']})
    write_json(root/'STATUS.json', {'M0': 'completed: original checkpoints/source and real generation parity verified',
        'M1': 'completed experiment; no gain over fixed vision; only 3 helpful held-out endpoint cases',
        'M2': 'completed: hard query-branch wrapper, fixed64, availability/future masking, real endpoint parity',
        'M3': 'completed bounded AVQA short-clip chronological/robustness experiment; long-video generalization untested',
        'M4': 'NOT_STARTED: audio selection efficacy criterion not met; no real IMU/Gaze/GUI integration',
        'M5': 'NOT_STARTED: second VLM not tested',
        'human_correspondence_annotation_required': False, 'original_checkpoints_overwritten': False})
    plot_summary(root, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=REPO/'artifacts/utility_gate/gate2_20261027')
    parser.add_argument('--data', type=Path, default=DATA_ROOT/'utility_gate/gate2_20261027')
    args = parser.parse_args()
    result = aggregate(args.root, args.data)
    print(json.dumps({'conclusion': result['conclusion'], 'failure_counts': result['failure_counts'],
                      'audit_tests': result['audit']['tests']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
