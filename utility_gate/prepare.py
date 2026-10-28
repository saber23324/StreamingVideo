"""Freeze answer-independent development subsets from already local QA data."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from .common import DATA_ROOT, REPO, fingerprint, sha256, validate_splits, write_json, write_jsonl


def ranked(values, seed, namespace):
    return sorted(values, key=lambda value: fingerprint([seed, namespace, str(value)]))


def prepare(output, train_videos=80, val_videos=20, test_videos=40, questions_per_video=2, seed=1234):
    output = Path(output)
    if output.exists():
        raise FileExistsError('use a fresh data directory; split and selection rules are immutable')
    manifest_path = REPO / 'results/dataset_manifests/streamingbench_realtime.json'
    old_split_path = REPO / 'results/dataset_splits/streamingbench_realtime_seed1234_80_20.json'
    manifest = json.loads(manifest_path.read_text())
    old = json.loads(old_split_path.read_text())
    groups = defaultdict(list)
    for row in manifest['rows']:
        if not Path(row['video_path']).is_file():
            raise FileNotFoundError(row['video_path'])
        groups[row['videoID']].append(row)
    development = ranked(old['train'], seed, 'development-video-order')
    heldout = ranked(old['test'], seed, 'heldout-video-order')
    if len(development) < train_videos + val_videos or len(heldout) < test_videos:
        raise ValueError('insufficient existing split; never borrow held-out videos for training')
    split = {'train': development[:train_videos], 'val': development[train_videos:train_videos+val_videos],
             'test': heldout[:test_videos]}
    selected = []
    for subset, ids in split.items():
        for video in ids:
            # Fixed earliest-Q rule uses question timestamps, never answers or branch results.
            for row in sorted(groups[video], key=lambda r: (r['query_time_seconds'], r['question_id']))[:questions_per_video]:
                selected.append({**row, 'dataset': 'streamingbench_rtvu_development', 'split': subset,
                                 'source_video_id': 'streamingbench:' + video,
                                 'question_modality': 'visual', 'media_sha256': sha256(row['video_path'])})
    audit = validate_splits(selected)
    output.mkdir(parents=True)
    write_jsonl(output / 'qa_manifest.jsonl', selected)
    write_json(output / 'split_manifest.json', {'seed': seed, **split,
        'protocol': 'internal development split, NOT the official StreamingBench evaluation',
        'source_manifest_sha256': sha256(manifest_path), 'source_split_sha256': sha256(old_split_path),
        'selection': 'hash-ranked original video groups, first two timestamp-sorted questions per video',
        'questions_per_video': questions_per_video, 'source_audit': audit,
        'counts': dict(Counter(r['split'] for r in selected)),
        'dataset_source': 'https://huggingface.co/datasets/mjuicem/StreamingBench',
        'license': 'no explicit dataset license in the already-local dataset card; no redistribution or new media download',
        'limitations': ['source IDs/paths/byte duplicates checked; original pretraining corpus is not fully known',
                       'first-Q sampling changes task/duration distribution',
                       'earlier owner development used StreamingBench; only the two VideoMME-trained routers are loaded']})
    # M4 has an observation schedule independent of future questions: 2 s cadence.
    streaming = []
    for subset in ('train', 'val', 'test'):
        candidates = [r for r in selected if r['split'] == subset and 2 <= r['query_time_seconds'] <= 24]
        unique = {}
        for row in sorted(candidates, key=lambda r: (r['query_time_seconds'], r['question_id'])):
            unique.setdefault(row['source_video_id'], row)
        quota = {'train': 24, 'val': 12, 'test': 12}[subset]
        for source in ranked(unique, seed, 'streaming-credit-subset')[:quota]:
            streaming.append({**unique[source], 'experiment': 'causal_segment', 'frame_interval_s': 2.0})
    validate_splits(streaming)
    write_jsonl(output / 'streaming_manifest.jsonl', streaming)
    write_json(output / 'preparation.json', {'primary': audit,
        'streaming_counts': dict(Counter(r['split'] for r in streaming)),
        'manual_annotation_required': False, 'new_media_downloads': 0})
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DATA_ROOT / 'utility_gate/run_20261026')
    parser.add_argument('--train-videos', type=int, default=80)
    parser.add_argument('--val-videos', type=int, default=20)
    parser.add_argument('--test-videos', type=int, default=40)
    parser.add_argument('--questions-per-video', type=int, default=2)
    args = parser.parse_args()
    rows = prepare(args.output, args.train_videos, args.val_videos, args.test_videos, args.questions_per_video)
    print(json.dumps({'questions': len(rows), 'output': str(args.output)}, ensure_ascii=False))


def prepare_cross_domain(output, primary_manifest, seed=1234):
    """Lock small transfer sets before fitting: no transfer labels enter selection."""
    from datasets import load_dataset
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows = []
    mlvu_path = REPO / 'results/dataset_manifests/mlvu_full_mcqa.json'
    split_path = REPO / 'results/dataset_splits/mlvu_full_mcqa_seed1234_pathgroup_80_20.json'
    mlvu = json.loads(mlvu_path.read_text())['rows']
    heldout = set(json.loads(split_path.read_text())['test'])
    chosen_paths = set()
    for task in sorted({r['task_type'] for r in mlvu}):
        candidates = {str(Path(r['video_path']).resolve()) for r in mlvu if r['task_type'] == task and r['videoID'] in heldout}
        chosen_paths.update(ranked(candidates, seed, 'mlvu-transfer-'+task)[:2])
    for row in mlvu:
        if str(Path(row['video_path']).resolve()) in chosen_paths:
            rows.append({**row, 'dataset': 'mlvu_dev_transfer_subset', 'split': 'test',
                'source_video_id': 'mlvu:' + Path(row['video_path']).stem,
                'question_modality': 'visual', 'media_sha256': sha256(row['video_path'])})
    videomme = load_dataset('lmms-lab/Video-MME', 'videomme', split='test', cache_dir='/home/yxd/.cache/huggingface')
    old = json.loads((REPO / 'results/adbt_videomme_split.json').read_text())
    for duration in ('short', 'medium', 'long'):
        candidates = {r['videoID'] for r in videomme if r['videoID'] in old['test'] and r['duration'] == duration}
        chosen = set(ranked(candidates, seed, 'videomme-transfer-'+duration)[:3])
        for row in videomme:
            if row['videoID'] in chosen:
                path = Path('/home/yxd/.cache/huggingface/videomme/data') / (row['videoID'] + '.mp4')
                if not path.is_file():
                    raise FileNotFoundError(path)
                rows.append({**row, 'question_id': 'videomme:' + row['question_id'],
                    'video_path': str(path.resolve()), 'query_time_seconds': None,
                    'dataset': 'videomme_owner_heldout_transfer_subset', 'split': 'test',
                    'source_video_id': 'youtube:' + row['videoID'], 'question_modality': 'visual',
                    'media_sha256': sha256(path)})
    primary = [json.loads(line) for line in Path(primary_manifest).read_text().splitlines() if line]
    validate_splits(primary + rows)
    output.mkdir(parents=True)
    write_jsonl(output / 'qa_manifest.jsonl', rows)
    write_json(output / 'selection.json', {'questions': len(rows), 'datasets': dict(Counter(r['dataset'] for r in rows)),
        'selection': 'hash-ranked held-out source videos; two per MLVU task, three per VideoMME duration',
        'tuning_allowed': False, 'source_manifest_sha256': sha256(mlvu_path), 'source_split_sha256': sha256(split_path),
        'protocol': 'eight-frame transfer diagnostic, NOT published benchmark accuracy',
        'license': 'reuse already-local datasets; original cards/terms retained in existing snapshots',
        'source_urls': ['https://github.com/JUNJIE99/MLVU', 'https://video-mme.github.io/home_page.html']})
    return rows


if __name__ == '__main__':
    main()
