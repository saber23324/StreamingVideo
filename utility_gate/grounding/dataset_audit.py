"""Metadata coverage, immutable plans and protection of historical Gate2."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from utility_gate.common import (AUDIO_ROUTER, DATA_ROOT, REPO, VISION_ROUTER,
                                 fingerprint, read_jsonl, sha256, write_json, write_jsonl)
from .llp_parser import clip_identity, read_intervals
from .spatial_mapping import LAYOUT

ROOT = REPO/'artifacts/utility_gate/gate2_grounding_20261027'
OLD = REPO/'artifacts/utility_gate/gate2_20261027'


def read(path): return json.loads(Path(path).read_text())


def protected_assets():
    prefix = read(OLD/'qualified_model/selection.json')['primary']
    temporal = read(OLD/'temporal_calibration/streaming_threshold.json')
    return {'audio_router': str(AUDIO_ROUTER), 'vision_router': str(VISION_ROUTER),
            'prefix_gate': prefix['checkpoint'], 'prefix_threshold': prefix['threshold_file'],
            'temporal_gate': temporal['model_checkpoint'],
            'temporal_threshold': str(OLD/'temporal_calibration/streaming_threshold.json')}


def all_exclusions():
    previous = read_jsonl(DATA_ROOT/'utility_gate/gate2_20261027/qa_manifest.jsonl')
    fit = {r['videoID'][:11] for r in previous if r['split'] != 'test'}
    seen = {r['videoID'][:11] for r in previous}
    owner = read(REPO/'results/adbt_videomme_split.json')
    # Exclude both owner train/evaluation sources for conservative independence.
    baseline = {str(v) for split in ('train', 'test') for v in owner[split]}
    return previous, fit, seen, baseline


def official_vggss():
    rows = read(DATA_ROOT/'vggss/annotations/raw/vggss.json')
    seen = set()
    for row in rows:
        if set(row) != {'file', 'class', 'bbox'} or row['file'] in seen:
            raise ValueError('official VGGSS JSON schema/identity mismatch')
        from utility_gate.gate2_data import original_source
        original_source(row['file'])
        seen.add(row['file'])
    return rows


def coverage_and_plan(root=ROOT):
    root = Path(root)
    old, fit, seen, baseline = all_exclusions()
    vgg = official_vggss()
    vgg_ids, vgg_sources = {r['file'] for r in vgg}, {r['file'][:11] for r in vgg}
    local = {p.stem: str(p) for p in (DATA_ROOT/'avqa/videos').glob('*.mp4')}
    media = read(DATA_ROOT/'avqa/acquisition_20261027/media_repository.json')
    mirror = {Path(x['rfilename']).stem for x in media['siblings'] if x['rfilename'].startswith('videos/')}
    llp_dir = DATA_ROOT/'llp_avvp/official_repo/data'
    rejected = []
    audio = read_intervals(llp_dir/'AVVP_eval_audio.csv', strict=False, rejections=rejected)
    visual = read_intervals(llp_dir/'AVVP_eval_visual.csv', strict=False, rejections=rejected)
    bad_label_names = {r['filename'] for r in rejected}
    llp_meta = read(DATA_ROOT/'llp_avvp/mirror_UnFaZeD07.json')
    if llp_meta['gated'] or llp_meta.get('disabled'):
        raise ValueError('LLP media mirror must be anonymous public')
    llp_media = {Path(x['rfilename']).stem: x['rfilename'] for x in llp_meta['siblings'] if x['rfilename'].startswith('videos/')}
    full_llp = list(csv.DictReader((llp_dir/'AVVP_dataset_full.csv').open(), delimiter='\t'))
    llp_source_counts = Counter()
    for row in full_llp:
        try: llp_source_counts[clip_identity(row['filename'])[0]] += 1
        except ValueError: pass
    official_split, llp_splits = {}, {}
    for split in ('val', 'test'):
        rows = list(csv.DictReader((llp_dir/f'AVVP_{split}_pd.csv').open(), delimiter='\t'))
        llp_splits[split] = rows
        for r in rows:
            official_split[r['filename']] = split
    llp_all = list(official_split)
    llp_candidates = []
    for name in llp_all:
        if name in bad_label_names: continue
        source, start, _ = clip_identity(name)
        canonical = source+f'_{start:06d}'
        if official_split[name] == 'test' and source not in fit|baseline and llp_source_counts[source] == 1:
            if canonical in local or source in llp_media:
                llp_candidates.append({'clip_id': name, 'canonical_clip_id': canonical, 'source_video_id': source,
                    'clip_offset_s': start, 'official_split': 'test', 'media_path': local.get(canonical),
                    'mirror_path': llp_media.get(source), 'audio_events': audio.get(name, []),
                    'visual_events': visual.get(name, []), 'same_source_multiple_offsets': False})
    llp_selected = sorted(llp_candidates, key=lambda r: fingerprint([1234, 'grounding-llp', r['clip_id']]))[:128]
    llp_selected_sources = {r['source_video_id'] for r in llp_selected}
    eligible_vgg = [r for r in vgg if r['file'] in mirror and r['file'][:11] not in fit|baseline]
    eligible_vgg = sorted(eligible_vgg, key=lambda r: (r['file'] not in local, fingerprint([1234, 'grounding-vggss', r['file']])))
    vgg_selected, used = [], set()
    for row in eligible_vgg:
        source = row['file'][:11]
        if source in used: continue
        used.add(source)
        vgg_selected.append({**row, 'clip_id': row['file'], 'source_video_id': source,
            'clip_offset_s': int(row['file'][12:]), 'media_path': local.get(row['file']),
            'annotation_time_s': 5., 'time_provenance': 'Chen21 section4.1: center frame of nominal10s VGGSound clip',
            'annotation_tolerance_s': .1, 'evaluation_mode': 'annotated_frame_spatial_diagnostic'})
        if len(vgg_selected) == 128: break
    avqa_eval = read(DATA_ROOT/'avqa/official_source/data/annotation/val_qa.json')
    avqa_train = read(DATA_ROOT/'avqa/official_source/data/annotation/train_qa.json')
    zip_index = read(DATA_ROOT/'avqa/Vynce_zip_index.json') if (DATA_ROOT/'avqa/Vynce_zip_index.json').exists() else None
    zip_media = {Path(r['path']).stem: r['path'] for r in zip_index['files'] if r['path'].endswith('.mp4')} if zip_index else {}
    # Held-out QA evaluation is source-disjoint from all Gate fits, previous
    # inspected evaluations, owner baseline IDs, VGGSS and selected LLP.
    excluded = seen|baseline|vgg_sources|llp_selected_sources
    candidates = [r for r in avqa_eval if r['video_name'][:11] not in excluded
                  and (r['video_name'] in mirror or r['video_name'] in zip_media)]
    rare = [r for r in candidates if r['question_relation'] != 'Both']
    both = sorted([r for r in candidates if r['question_relation'] == 'Both'],
                  key=lambda r: fingerprint([1234, 'grounding-avqa-both', r['id']]))[:192]
    avqa_selected = [{'record': r, 'source_video_id': r['video_name'][:11],
                     'media_path': local.get(r['video_name']), 'zip_member': zip_media.get(r['video_name']),
                     'mirror_available': r['video_name'] in mirror} for r in rare+both]
    coverage = {'created_utc': datetime.now(timezone.utc).isoformat(),
        'historical_negative_result_preserved': True, 'no_new_training': True,
        'vggss': {'official_annotations': len(vgg), 'classes': len({r['class'] for r in vgg}),
            'exact_AVQA_local': len(vgg_ids&local.keys()),
            'exact_local_excluding_gate_fit': sum(r[:11] not in fit for r in vgg_ids&local.keys()),
            'exact_pinned_public_mirror': len(vgg_ids&mirror), 'eligible': len(eligible_vgg), 'planned': len(vgg_selected),
            'annotation_time_verified_from_paper': True, 'frame_time_s': 5.},
        'llp': {'official_dense_split_videos': len(llp_all), 'audio_annotated_videos': len(audio),
            'visual_annotated_videos': len(visual), 'audio_rows': sum(map(len, audio.values())),
            'visual_rows': sum(map(len, visual.values())),
            'missing_dense_rows_interpretation': 'empty event set within the official dense val/test universe',
            'rejected_dense_rows': rejected, 'entire_clip_excluded_if_any_dense_row_malformed': True,
            'local_exact': sum((clip_identity(n)[0]+f'_{clip_identity(n)[1]:06d}') in local for n in llp_all if n not in bad_label_names),
            'public_mirror_clipped_files': len(llp_media), 'eligible_test_candidates': len(llp_candidates), 'planned': len(llp_selected),
            'mirror_revision': llp_meta['sha'], 'source_offsets_unique_only': True},
        'avqa': {'official_train_QA': len(avqa_train), 'official_eval_QA': len(avqa_eval),
            'all_eval_by_relation': dict(Counter(r['question_relation'] for r in avqa_eval)),
            'local_eval_by_relation': dict(Counter(r['question_relation'] for r in avqa_eval if r['video_name'] in local)),
            'eligible_new_by_relation': dict(Counter(r['question_relation'] for r in candidates)),
            'planned_by_relation': dict(Counter(r['record']['question_relation'] for r in avqa_selected)),
            'new_sources_exclude_old_train_val_test_vggss_owner_and_selected_llp': True,
            'zip_index_available': zip_index is not None},
        'layout': LAYOUT, 'exclusion_sizes': {'gate_fit_sources': len(fit), 'all_previous_gate2_sources': len(seen),
            'owner_videomme_exclusion_list': len(baseline), 'vggss_all_sources_for_new_avqa': len(vgg_sources)},
        'assets': {k: {'path': p, 'sha256': sha256(p)} for k, p in protected_assets().items()},
        'official_sources': {name: subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=path, text=True).strip()
            for name, path in [('vggss_code', DATA_ROOT/'vggss/official_source'), ('llp', DATA_ROOT/'llp_avvp/official_repo')]},
        'annotation_sha256': {str(p): sha256(p) for p in [DATA_ROOT/'vggss/annotations/raw/vggss.json',
             llp_dir/'AVVP_eval_audio.csv', llp_dir/'AVVP_eval_visual.csv', llp_dir/'AVVP_test_pd.csv',
             DATA_ROOT/'avqa/official_source/data/annotation/val_qa.json']}}
    write_json(root/'datasets/coverage.json', coverage)
    for name, rows in [('vggss', vgg_selected), ('llp', llp_selected), ('avqa', avqa_selected)]:
        path = root/'datasets'/(name+'_plan.jsonl')
        if path.exists() and read_jsonl(path) != rows:
            raise ValueError('answer-independent plan already locked; do not replace')
        write_jsonl(path, rows)
    write_json(root/'M0_metadata.json', coverage)
    return coverage


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT)
    args = p.parse_args()
    print(json.dumps(coverage_and_plan(args.output), ensure_ascii=False))


if __name__ == '__main__': main()
