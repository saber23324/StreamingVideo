"""M0/M4 paired, resumable, same-prefix frozen QA inference."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import torch

from .common import (REPO, append_jsonl, fingerprint, read_jsonl, seed_all, sha256,
                     validate_splits, write_json, write_jsonl)


def run(manifest, output, cache_dir, mode='prefix', max_frames=8, shard=0, shards=1):
    from .runtime import FrozenRuntime, score_answer
    rows = read_jsonl(manifest)
    validate_splits(rows)
    source_ids = sorted({r['source_video_id'] for r in rows})
    selected = {value for i, value in enumerate(source_ids) if i % shards == shard}
    rows = [r for r in rows if r['source_video_id'] in selected]
    output, cache_dir = Path(output), Path(cache_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    seed_all(1234)
    if mode == 'streaming':
        from .strict_runtime import StrictStreamingRuntime
        runtime = StrictStreamingRuntime()
    else:
        runtime = FrozenRuntime()
    config = {'manifest_sha256': sha256(manifest), 'runtime': runtime.metadata,
              'mode': mode, 'max_frames': max_frames, 'seed': 1234}
    version = fingerprint(config)
    config_path = output / 'paired_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != {**config, 'fingerprint': version}:
        raise ValueError('resume configuration changed; use a fresh output directory')
    write_json(config_path, {**config, 'fingerprint': version})
    regression = runtime.regression_fixture(REPO / 'artifacts/relevance_gate/m0-gate-disabled-final-20261022')
    write_json(output / f'regression_shard{shard}.json', regression)
    result_path = output / f'paired_qa.shard{shard}.jsonl'
    previous = read_jsonl(result_path) if result_path.is_file() else []
    done = {r['question_id'] for r in previous}
    if len(done) != len(previous) or any(r['config_fingerprint'] != version for r in previous):
        raise ValueError('duplicate or stale resume records')
    for index, row in enumerate(rows, 1):
        if row['question_id'] in done:
            continue
        key = fingerprint({'media_sha256': row['media_sha256'], 'source_video_id': row['source_video_id'],
            'query_timestamp_s': row.get('query_time_seconds'), 'config': version})
        path = cache_dir / (key + '.pt')
        if path.exists():
            payload = torch.load(path, map_location='cpu', weights_only=True)
            if payload['cache_key'] != key:
                raise ValueError('stale tensor cache')
        else:
            if sha256(row['video_path']) != row['media_sha256']:
                raise ValueError('media bytes changed after split freeze')
            payload = runtime.encode(row, mode, max_frames)
            payload.update(cache_key=key, config_fingerprint=version,
                           source_video_id=row['source_video_id'], split=row['split'])
            temporary = path.with_suffix('.tmp')
            torch.save(payload, temporary)
            temporary.replace(path)
        prompt = runtime.prompt(row)
        branches = {}
        # Rotate order without inspecting labels, reducing a systematic first-call timing bias.
        names = ['audio', 'vision', 'full_vision']
        offset = int(fingerprint(row['question_id'])[:8], 16) % 3
        names = names[offset:] + names[:offset]
        for name in names:
            if name == 'full_vision':
                latents = runtime.full_vision(payload['visual'].to(runtime.device))
            else:
                latents = payload[name + '_latents'].to(runtime.device)
            generation = runtime.generate(prompt, latents)
            branches[name] = score_answer(generation, row['answer'], row['dataset'])
        if branches['audio']['visual_tokens'] != branches['vision']['visual_tokens']:
            raise AssertionError('unequal compressed branch budgets')
        record = {k: row[k] for k in ('question_id', 'source_video_id', 'videoID', 'video_path',
                  'dataset', 'split', 'task_type', 'answer', 'media_sha256')}
        record.update(query_timestamp_s=row.get('query_time_seconds'),
            question_modality=row.get('question_modality', 'unknown'),
            question_sha256=fingerprint({'question': row['question'], 'options': row['options']}),
            prompt_sha256=fingerprint(prompt.tolist()), config_fingerprint=version,
            feature_cache_key=key, feature_cache_path=str(path.resolve()), mode=mode,
            timestamps_s=payload['timestamps'].tolist(), frame_pts_s=payload['frame_pts_s'],
            audio_quality=payload['audio_quality'], audio_valid=payload['audio_valid'].tolist(),
            audio_missing_fraction=1-float(payload['audio_valid'].float().mean()),
            token_budget_per_frame=64, timing=payload['timing'], branches=branches,
            accuracy_utility=int(branches['audio']['correct'])-int(branches['vision']['correct']),
            nll_utility=branches['vision']['gold_nll']-branches['audio']['gold_nll'])
        append_jsonl(result_path, record)
        print(json.dumps({'shard': shard, 'row': index, 'total': len(rows), 'id': row['question_id'],
            'split': row['split'], 'AV': branches['audio']['prediction'], 'V': branches['vision']['prediction'],
            'accuracy_utility': record['accuracy_utility'], 'nll_utility': round(record['nll_utility'], 4)}, ensure_ascii=False), flush=True)
    write_json(output / f'completed_shard{shard}.json', {'completed': len(rows), 'fingerprint': version,
        'all_parameters_frozen': all(not p.requires_grad for module in
            (runtime.model, runtime.audio_encoder, runtime.audio_router, runtime.vision_router) for p in module.parameters())})


def merge(manifest, output):
    output = Path(output)
    expected = {r['question_id'] for r in read_jsonl(manifest)}
    rows = [row for path in sorted(output.glob('paired_qa.shard*.jsonl')) for row in read_jsonl(path)]
    if len(rows) != len(expected) or {r['question_id'] for r in rows} != expected:
        raise ValueError('paired run incomplete or duplicate IDs')
    if len({r['config_fingerprint'] for r in rows}) != 1:
        raise ValueError('cannot combine different paired protocols')
    validate_splits(rows)
    write_jsonl(output / 'paired_qa.jsonl', sorted(rows, key=lambda r: r['question_id']))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--mode', choices=['prefix', 'streaming'], default='prefix')
    parser.add_argument('--max-frames', type=int, default=8)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--merge', action='store_true')
    args = parser.parse_args()
    if args.merge:
        print({'merged': len(merge(args.manifest, args.output))})
    else:
        if args.cache_dir is None or args.shards < 1 or not 0 <= args.shard < args.shards:
            parser.error('valid shard and --cache-dir required')
        run(args.manifest, args.output, args.cache_dir, args.mode, args.max_frames, args.shard, args.shards)


if __name__ == '__main__':
    main()
