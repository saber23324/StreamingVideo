"""New source-disjoint AVQA generation, frozen models, paired grouped reports."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path

import numpy as np
import torch

from utility_gate.common import (DATA_ROOT, append_jsonl, fingerprint, read_jsonl, seed_all,
    sha256, validate_splits, write_json, write_jsonl)
from utility_gate.features import GateFeatures, take_features
from utility_gate.gate2_interfaces import features_from_cache
from utility_gate.gate2_runtime import qualified_latents
from utility_gate.gate2_temporal import Gate2OnlineState, cached_stream
from utility_gate.runtime import score_answer
from utility_gate.strict_runtime import StrictStreamingRuntime
from .dataset_audit import ROOT, all_exclusions, official_vggss, read
from .native import encode_times
from .runtime import gates, lock_evaluation
from .stats import mean_ci


class MediaRuntime(StrictStreamingRuntime):
    def __init__(self):
        super().__init__(); self.external_audio = {}

    def causal_windows(self, video_path, timestamps):
        return super().causal_windows(self.external_audio.get(str(video_path), video_path), timestamps)


def validate_sources(rows, fit, seen, owner, vggss, llp):
    excluded = fit|seen|owner|vggss|llp
    for row in rows:
        source = row['source_video_id'].removeprefix('youtube:')
        if source in excluded or row['official_split'] != 'evaluation' or row['split'] != 'test':
            raise ValueError('non-independent or non-evaluation QA source')
    validate_splits(rows)


def prefix_features(payload):
    # Only native observation tensors reach GateFeatures; relation/QA/gold are
    # deliberately absent, even if a caller attaches them to the cache dict.
    observed = {k: payload[k] for k in ('visual', 'causal_audio', 'timestamps', 'audio_valid')}
    observed['audio_valid'] = observed['audio_valid'].to(observed['visual'].device)
    value = features_from_cache(observed, 'anchor_history')
    current = take_features(value, slice(-1, None))
    return GateFeatures(current.visual, current.audio, current.context, payload['audio_valid'].any().reshape(1))


def random_masks(policies):
    """Exact available-rate controls; time strata use only same-time decisions.

    Stream control is a cohort diagnostic: it consumes same-time activation
    COUNT, never another stream's event labels or future decisions. It is not
    a standalone single-stream deployment policy.
    """
    sorted_rows = sorted(policies, key=lambda r: r['question_id'])
    rng = np.random.default_rng(1234)
    eligible = [r['question_id'] for r in sorted_rows if any(r['available'])]
    k = sum(r['prefix_accepted'] for r in sorted_rows)
    chosen = set(rng.choice(eligible, k, replace=False).tolist()) if k else set()
    prefix = {r['question_id']: r['question_id'] in chosen for r in sorted_rows}
    stream = {r['question_id']: [False]*len(r['times']) for r in sorted_rows}
    for time in sorted({t for r in sorted_rows for t in r['times']}):
        eligible = []; count = 0
        for row in sorted_rows:
            if time not in row['times']: continue
            j = row['times'].index(time)
            if row['available'][j]: eligible.append((row['question_id'], j))
            count += row['stream_selected'][j]
        generator = np.random.default_rng(int(fingerprint([1234, 'same-time-matched-random', time])[:8], 16))
        selected = generator.choice(len(eligible), count, replace=False) if count else []
        for i in selected:
            q, j = eligible[int(i)]; stream[q][j] = True
    assert sum(prefix.values()) == k
    assert sum(sum(v) for v in stream.values()) == sum(sum(r['stream_selected']) for r in policies)
    return {'prefix': prefix, 'stream': stream,
            'prefix_rule': 'exact global prefix count, available prefixes only, whole-prefix feasibility diagnostic',
            'stream_rule': 'exact per-timestamp cohort activation count and available candidates, no future-time masks',
            'not_standalone_single_stream_deployment_policy': True, 'uses_gold_answers': False}


def interval(values, sources):
    result = mean_ci(values, sources)
    if result['sources'] == 1:
        result['bootstrap_ci95_diagnostic'] = result['ci95']
        result.update(ci95=None, reason='one independent source cannot estimate a population confidence interval')
    return result


def group_summary(rows):
    if not rows: return {'n': 0, 'sources': 0, 'status': 'NO_AVAILABLE_INDEPENDENT_MEDIA'}
    sources = [r['source_video_id'] for r in rows]
    methods = {}
    for method in rows[0]['branches']:
        correct = [r['branches'][method]['correct'] for r in rows]
        delta = [int(r['branches'][method]['correct'])-int(r['branches']['vision_only_64']['correct']) for r in rows]
        nll = [r['branches'][method]['gold_nll'] for r in rows]
        nll_delta = [r['branches']['vision_only_64']['gold_nll']-r['branches'][method]['gold_nll'] for r in rows]
        methods[method] = {'correct': sum(correct), 'n': len(rows), 'accuracy': interval(correct, sources),
            'gold_letter_nll': interval(nll, sources), 'paired_accuracy_minus_vision': interval(delta, sources),
            'paired_NLL_gain_over_vision': interval(nll_delta, sources)}
    quadrants = Counter()
    for r in rows:
        a = r['branches']['qualified_audio_64']['correct']; v = r['branches']['vision_only_64']['correct']
        quadrants['both_correct' if a and v else 'only_audio_correct' if a else 'only_vision_correct' if v else 'both_wrong'] += 1
    selection = {}
    for method in ('gate2_prefix_64', 'gate2_stream_64'):
        selection[method] = {'beneficial_missed': sum(r['branches']['qualified_audio_64']['correct'] and
            not r['branches']['vision_only_64']['correct'] and not r['branches'][method]['correct'] for r in rows),
            'harmful_avoided': sum(r['branches']['vision_only_64']['correct'] and not r['branches']['qualified_audio_64']['correct']
            and r['branches'][method]['correct'] for r in rows),
            'new_harm_to_correct_vision': sum(r['branches']['vision_only_64']['correct'] and not r['branches'][method]['correct'] for r in rows)}
    return {'n': len(rows), 'sources': len(set(sources)), 'methods': methods, 'qualified_complementarity': dict(quadrants),
        'offline_two_endpoint_oracle_correct': sum(r['branches']['qualified_audio_64']['correct'] or r['branches']['vision_only_64']['correct'] for r in rows),
        'selection': selection, 'prefix_on_count': sum(r['policy']['prefix_accepted'] for r in rows),
        'random_prefix_on_count': sum(r['random_prefix_accepted'] for r in rows),
        'stream_on_count': sum(sum(r['policy']['stream_selected']) for r in rows),
        'random_stream_on_count': sum(sum(r['random_stream_selected']) for r in rows),
        'windows': sum(len(r['policy']['times']) for r in rows),
        'available_windows': sum(sum(r['policy']['available']) for r in rows)}


def run(root=ROOT):
    root = Path(root); output = root/'avqa'; output.mkdir(exist_ok=True)
    rows = read_jsonl(root/'datasets/avqa_manifest.jsonl')
    supplement = root/'datasets/avqa_supplement_manifest.jsonl'
    if supplement.exists(): rows += read_jsonl(supplement)
    rows.sort(key=lambda r: r['question_id'])
    old, fit, seen, owner = all_exclusions()
    validate_sources(rows, fit, seen, owner, {r['file'][:11] for r in official_vggss()},
                     {r['source_video_id'] for r in read_jsonl(root/'datasets/llp_plan.jsonl')})
    # Byte duplicates against protected training/evaluation media are excluded,
    # beyond source-id checks. Exact new duplicates may share a TEST source.
    old_hashes = {r['media_sha256'] for r in old}
    if any(r['media_sha256'] in old_hashes for r in rows): raise ValueError('old media byte duplicate entered new cohort')
    lock_evaluation(root, 'avqa')
    extra_lock = {'supplement_sha256': sha256(supplement) if supplement.exists() else None,
        'supplement_plan_sha256': sha256(root/'datasets/avqa_supplement_plan.jsonl') if supplement.exists() else None,
        'combined_ids': [r['question_id'] for r in rows], 'before_generation': True, 'no_outcome_selection': True}
    p = root/'pretest_lock_avqa_supplement.json'
    if p.exists() and read(p) != extra_lock: raise ValueError('new QA cohort already locked')
    write_json(p, extra_lock)
    seed_all(1234); runtime = MediaRuntime()
    prefix_gate, prefix_tau, temporal_gate, temporal_tau = gates(runtime.device)
    for r in rows:
        if not r['has_audio_track'] and r.get('external_audio_path'):
            runtime.external_audio[str(r['media_path'])] = r['external_audio_path']
    cache = DATA_ROOT/'utility_gate/gate2_grounding_20261027/native_cache'; cache.mkdir(parents=True, exist_ok=True)
    policy_path = output/'policies.jsonl'
    policies = read_jsonl(policy_path) if policy_path.exists() else []
    policy_map = {r['question_id']: r for r in policies}
    with torch.inference_mode():
        for i, row in enumerate(rows):
            if row['question_id'] in policy_map: continue
            try:
                times = [float(t) for t in range(2, 11, 2) if t <= row['query_time_seconds']+1e-6]
                payload, _ = encode_times(runtime, row['media_path'], times)
                features = prefix_features(payload)
                pred = prefix_gate.predict(features.to(runtime.device))
                accept = bool(pred.valid_audio[0] and pred.utility_score[0] > prefix_tau)
                state = cached_stream(runtime, temporal_gate, temporal_tau, payload)
                path = cache/(row['question_id'].replace(':', '_')+'.pt')
                torch.save({k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in payload.items() if k not in ('audio_P', 'vision_P')}, path)
                record = {'question_id': row['question_id'], 'source_video_id': row['source_video_id'],
                    'times': times, 'available': payload['audio_valid'].tolist(), 'quality': payload['audio_quality'],
                    'prefix_accepted': accept, 'prefix_score': float(pred.utility_score[0]), 'prefix_threshold': prefix_tau,
                    'stream_selected': [t['accepted'] for t in state.trace], 'stream_trace': state.trace,
                    'cache_path': str(path), 'cache_sha256': sha256(path), 'QA_inputs_to_gate': False}
                append_jsonl(policy_path, record); policies.append(record); policy_map[row['question_id']] = record
            except Exception as error:
                append_jsonl(output/'failures.jsonl', {'question_id': row['question_id'], 'stage': 'encode', 'reason': str(error)})
            if (i+1) % 16 == 0: print({'encoded_new_QA': i+1}, flush=True)
    random = random_masks(policies); write_json(output/'random_masks_before_QA.json', random)
    result_path = output/'per_qa.jsonl'
    done = {r['question_id'] for r in read_jsonl(result_path)} if result_path.exists() else set()
    silent = runtime.audio_encoder.encode_windows(torch.zeros(1, 32000), runtime.device).to(runtime.device).float()
    with torch.inference_mode():
        for i, row in enumerate(rows):
            q = row['question_id']
            if q in done or q not in policy_map: continue
            try:
                policy = policy_map[q]
                payload = torch.load(policy['cache_path'], map_location=runtime.device, weights_only=True)
                qualified, _ = qualified_latents(runtime, payload); qualified = qualified[0]
                stream = cached_stream(runtime, temporal_gate, temporal_tau, payload)
                assert [t['accepted'] for t in stream.trace] == policy['stream_selected']
                random_stream = cached_stream(runtime, temporal_gate, temporal_tau, payload, force=random['stream'][q])
                null, _ = runtime.audio_router(payload['visual'], silent.expand(len(payload['visual']), -1, -1), payload['timestamps'])
                latents = {'vision_only_64': payload['vision_latents'], 'legacy_audio_64': payload['audio_latents'],
                    'qualified_audio_64': qualified, 'full_vision_196': payload['visual'], 'stage3_null_audio_64': null,
                    'gate2_prefix_64': qualified if policy['prefix_accepted'] else payload['vision_latents'],
                    'gate2_stream_64': stream.bank(),
                    'random_matched_prefix_64': qualified if random['prefix'][q] else payload['vision_latents'],
                    'random_matched_stream_64': random_stream.bank()}
                results = {}; generated = []
                prompt = runtime.prompt(row)
                for name, tensor in latents.items():
                    identical = next((other for other, existing in generated if torch.equal(tensor, existing)), None)
                    if identical is None:
                        results[name] = score_answer(runtime.generate(prompt, tensor), row['answer'], row['dataset'])
                        results[name]['generation_reused_from'] = None; generated.append((name, tensor))
                    else:
                        results[name] = {**results[identical], 'generation_reused_from': identical,
                                         'identical_visual_token_input_verified': True}
                result = {'question_id': q, 'source_video_id': row['source_video_id'], 'videoID': row['videoID'],
                    'question_modality': row['question_modality'], 'question': row['question'], 'options': row['options'],
                    'gold_answer': row['answer'], 'media_sha256': row['media_sha256'], 'video_path': row['video_path'],
                    'policy': policy, 'random_prefix_accepted': random['prefix'][q], 'random_stream_selected': random['stream'][q],
                    'branches': results, 'QA_only_passed_to_VLM_after_routing': True,
                    'audio_source': runtime.external_audio.get(str(row['media_path']), row['media_path'])}
                append_jsonl(result_path, result)
            except Exception as error:
                append_jsonl(output/'failures.jsonl', {'question_id': q, 'stage': 'generation', 'reason': str(error)})
            if (i+1) % 16 == 0: print({'generated_new_QA': i+1}, flush=True)
    results = read_jsonl(result_path) if result_path.exists() else []
    summary = {'planned_media_QA': len(rows), 'evaluated': len(results), 'all': group_summary(results),
        'by_relation': {relation: group_summary([r for r in results if r['question_modality'] == relation]) for relation in ('View', 'Sound', 'Both')},
        'quality_counts': dict(Counter(q for r in results for q in r['policy']['quality'])),
        'new_training_or_threshold_calibration': False, 'new_sources_disjoint_from_previous_Gate2_and_VGGSS': True,
        'random_control': {k: v for k, v in random.items() if k not in ('prefix', 'stream')},
        'same_token_generations_reused_only_after_torch_equal': True,
        'scope': 'fixed source-disjoint AVQA subset; not the full official evaluation benchmark'}
    write_json(output/'summary.json', summary)
    with (output/'by_relation.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['relation', 'n', 'sources', 'method', 'correct', 'accuracy', 'delta_vs_vision', 'ci95'])
        writer.writeheader()
        for relation, group in summary['by_relation'].items():
            for method, value in group.get('methods', {}).items():
                writer.writerow({'relation': relation, 'n': group['n'], 'sources': group['sources'], 'method': method,
                    'correct': value['correct'], 'accuracy': value['accuracy']['mean'],
                    'delta_vs_vision': value['paired_accuracy_minus_vision']['mean'], 'ci95': value['paired_accuracy_minus_vision']['ci95']})
    failures = [r for r in results if r['branches']['vision_only_64']['correct'] and
                (not r['branches']['gate2_stream_64']['correct'] or not r['branches']['legacy_audio_64']['correct'])]
    write_jsonl(output/'negative_transfer_cases.jsonl', failures)
    print({'evaluated': len(results), 'relations': {k: v['n'] for k, v in summary['by_relation'].items()}})


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, default=ROOT)
    run(p.parse_args().output)


if __name__ == '__main__': main()
