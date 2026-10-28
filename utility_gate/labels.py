"""M0 complementarity and M1 automatic QA-utility labels, with no human labels."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np

from .common import fingerprint, read_jsonl, sha256, validate_splits, write_json, write_jsonl


def complementarity(rows):
    names = {(True, True): 'both_correct', (False, True): 'audio_only_win',
             (True, False): 'vision_only_win', (False, False): 'both_wrong'}
    counts = Counter(names[(bool(r['branches']['vision']['correct']), bool(r['branches']['audio']['correct']))] for r in rows)
    total = len(rows)
    result = {key: counts[key] for key in names.values()}
    result.update(n=total,
        vision_accuracy=sum(r['branches']['vision']['correct'] for r in rows)/total if total else None,
        audio_accuracy=sum(r['branches']['audio']['correct'] for r in rows)/total if total else None,
        full_vision_accuracy=sum(r['branches']['full_vision']['correct'] for r in rows)/total if total else None,
        oracle_accuracy=(counts['both_correct']+counts['audio_only_win']+counts['vision_only_win'])/total if total else None,
        net_audio_accuracy_gain=(counts['audio_only_win']-counts['vision_only_win'])/total if total else None,
        oracle_headroom_over_audio=counts['vision_only_win']/total if total else None)
    return result


def verify_pair(row):
    if row['token_budget_per_frame'] != 64:
        raise ValueError('wrong token budget')
    expected = 64 * len(row['timestamps_s'])
    branches = row['branches']
    for name in ('audio', 'vision'):
        if branches[name]['visual_tokens'] != expected:
            raise ValueError('paired visual budgets mismatch')
        if not math.isfinite(branches[name]['gold_nll']) or branches[name]['gold_nll'] < 0:
            raise ValueError('nonfinite or negative gold NLL')
    if branches['audio']['nll_protocol'] != branches['vision']['nll_protocol']:
        raise ValueError('paired NLL protocols mismatch')
    if branches['audio']['total_input_tokens'] != branches['vision']['total_input_tokens']:
        raise ValueError('paired prompts/lengths mismatch')
    if row.get('query_timestamp_s') is not None:
        if max(row['timestamps_s']) > row['query_timestamp_s'] + 1e-6:
            raise ValueError('future observation in QA prefix')
    if any(p > t+1e-5 for p, t in zip(row['frame_pts_s'], row['timestamps_s'])):
        raise ValueError('frame PTS exceeds its decision time')


def build_labels(rows, delta_quantile=.25):
    validate_splits(rows)
    for row in rows:
        verify_pair(row)
    train = [r for r in rows if r['split'] == 'train']
    if not train or not 0 <= delta_quantile <= 1:
        raise ValueError('training examples and a valid delta quantile required')
    # The value is derived from train only; the percentile is a recorded protocol choice.
    delta = float(np.quantile([abs(r['nll_utility']) for r in train], delta_quantile))
    labels = []
    for row in rows:
        av, vision = row['branches']['audio'], row['branches']['vision']
        accuracy = int(av['correct'])-int(vision['correct'])
        nll = vision['gold_nll']-av['gold_nll']
        if accuracy != row['accuracy_utility'] or not math.isclose(nll, row['nll_utility'], abs_tol=1e-8):
            raise ValueError('stored QA-utility delta cannot be recomputed')
        state = 'audio_better' if accuracy == 1 else 'vision_better' if accuracy == -1 else 'ambiguous'
        labels.append({k: row[k] for k in ('question_id', 'source_video_id', 'videoID', 'video_path',
                       'media_sha256', 'dataset', 'split', 'task_type', 'query_timestamp_s',
                       'feature_cache_key', 'feature_cache_path', 'config_fingerprint', 'question_modality')})
        labels[-1].update(vision_correct=vision['correct'], audio_correct=av['correct'],
            vision_gold_nll=vision['gold_nll'], audio_gold_nll=av['gold_nll'],
            accuracy_utility=accuracy, nll_utility=nll, label_state=state,
            correctness_confidence='discordant_QA_outcome' if accuracy else 'tie_use_continuous_NLL_only',
            nll_preference=1 if nll > delta else -1 if nll < -delta else 0,
            nll_direction_agrees_with_correctness=None if not accuracy else accuracy*nll > 0,
            audio_missing_fraction=row['audio_missing_fraction'],
            parser_valid=av['parser_valid'] and vision['parser_valid'],
            token_budget_per_frame=64, original_gold_answer=row['answer'],
            label_provenance='frozen paired downstream QA; no semantic correspondence labels',
            supervision_scope='prefix proxy' if row['mode'] == 'prefix' else 'whole-prefix endpoints; never copied to frame labels')
    return labels, {'delta': delta, 'delta_quantile': delta_quantile, 'fit_split': 'train',
                    'binary_supervision': 'accuracy-discordant examples only',
                    'regression_supervision': 'continuous nll_utility on all finite paired QA examples'}


def export(paired, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(paired)
    labels, label_config = build_labels(rows)
    write_jsonl(output / 'utility_labels.jsonl', labels)
    write_json(output / 'label_config.json', label_config)
    summaries = []
    for split in ('train', 'val', 'test'):
        selected = [r for r in rows if r['split'] == split]
        if selected:
            summaries.append({'split': split, 'task_type': 'ALL', **complementarity(selected)})
            for task in sorted({r['task_type'] for r in selected}):
                summaries.append({'split': split, 'task_type': task, **complementarity([r for r in selected if r['task_type'] == task])})
    with (output / 'complementarity.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    stats = {'source_paired_sha256': sha256(paired), 'split_audit': validate_splits(rows),
        'label_config': label_config, 'human_review_needed': False, 'by_split': {},
        'nonfinite_nll': 0, 'annotation_task': 'downstream branch utility, not semantic AV correspondence'}
    for split in ('train', 'val', 'test'):
        selected = [r for r in labels if r['split'] == split]
        if not selected:
            continue
        stats['by_split'][split] = {'questions': len(selected), 'videos': len({r['source_video_id'] for r in selected}),
            'label_counts': dict(Counter(r['label_state'] for r in selected)),
            'parser_failures': sum(not r['parser_valid'] for r in selected),
            'all_audio_invalid': sum(r['audio_missing_fraction'] == 1 for r in selected),
            'mean_missing_window_fraction': float(np.mean([r['audio_missing_fraction'] for r in selected])),
            'nll_utility_quantiles': np.quantile([r['nll_utility'] for r in selected], [0,.1,.25,.5,.75,.9,1]).tolist(),
            'tasks': dict(Counter(r['task_type'] for r in selected))}
    write_json(output / 'label_statistics.json', stats)
    development = [r for r in rows if r['split'] in ('train', 'val')]
    decision = complementarity(development)
    enough = decision['audio_only_win'] >= 5 and decision['vision_only_win'] >= 5
    lines = ['# M0 decision: frozen branch complementarity', '',
        'Decision uses train/val only. Held-out outcomes are diagnostic and never tune the Gate.', '',
        f'Development questions: {len(development)}. Audio-only wins: {decision["audio_only_win"]}; '
        f'vision-only wins: {decision["vision_only_win"]}; both correct: {decision["both_correct"]}; both wrong: {decision["both_wrong"]}.', '',
        f'Always AV accuracy: {decision["audio_accuracy"]:.4f}; vision: {decision["vision_accuracy"]:.4f}; '
        f'oracle: {decision["oracle_accuracy"]:.4f}; headroom over AV: {decision["oracle_headroom_over_audio"]:.4f}.', '',
        'Proceed with bounded cached-feature Gate feasibility training. ' + (
            'Both directions of complementarity exist; out-of-video prediction remains unproven.' if enough else
            'WARNING: few discordant examples in at least one direction; useful signal and selection headroom may be limited. Do not enlarge training or claim gains from this result alone.'), '',
        'Controls: same frozen 7B VLM, K=64/frame, frame timestamps, prompt, greedy decoder, strict letter scorer, NLL protocol.', '',
        'Confound: separately trained stage-1 static vision queries vs stage-3 audio/time queries. '
        'This measures operational checkpoint selection, not an isolated causal effect of audio.', '',
        'Prefix experiment uses eight-frame capped legacy sampling and legacy routing audio. It is an offline proxy, '
        'not an online per-frame Gate. M4 uses a separate fixed time grid and strict causal audio.', '',
        'No correspondence annotation or human approval is a prerequisite.']
    (output / 'M0_decision.md').write_text('\n'.join(lines) + '\n')
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    stats = export(args.paired, args.output)
    print(json.dumps(stats['by_split'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
