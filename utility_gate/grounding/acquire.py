"""Acquire only preselected public media; retain every access/QC failure."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess

import av

from utility_gate.common import DATA_ROOT, read_jsonl, sha256, write_json, write_jsonl
from utility_gate.gate2_data import fetch_video, normalize
from .dataset_audit import ROOT, read


def inspect(path):
    with av.open(str(path)) as c:
        if not c.streams.video: raise ValueError('no_video_track')
        stream = c.streams.video[0]
        return {'duration_s': c.duration/av.time_base if c.duration is not None else float(stream.duration*stream.time_base),
                'has_audio_track': bool(c.streams.audio), 'width': stream.width, 'height': stream.height,
                'average_rate': float(stream.average_rate) if stream.average_rate else None}


def llp_video(row):
    meta = read(DATA_ROOT/'llp_avvp/mirror_UnFaZeD07.json')
    if row.get('media_path'):
        path = Path(row['media_path'])
    else:
        path = DATA_ROOT/'llp_avvp/videos'/(row['canonical_clip_id']+'.mp4')
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/UnFaZeD07/LLP/resolve/{meta['sha']}/{row['mirror_path']}?download=true"
        if not path.exists():
            temporary = path.with_suffix('.part')
            command = ['curl', '--fail', '--silent', '--show-error', '--location', '--retry', '2', '--retry-all-errors',
                       '--max-time', '90', '--max-filesize', str(64*1024*1024), '--output', str(temporary), url]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode: raise RuntimeError(result.stderr[-240:])
            temporary.replace(path)
            write_json(path.with_suffix('.source.json'), {'repository': meta['id'], 'revision': meta['sha'],
                'url': url, 'sha256': sha256(path), 'bytes': path.stat().st_size,
                'clip_offset_s': row['clip_offset_s'], 'offset_provenance': 'official full TSV unique source and mirror trim script'})
    details = inspect(path)
    if details['duration_s'] > 10.5 or details['duration_s'] < 5:
        raise ValueError('mirror_clip_duration_not_nominal10s')
    return path, sha256(path), details


def run(dataset, root=ROOT):
    root = Path(root)
    plan = read_jsonl(root/'datasets'/(dataset+'_plan.jsonl'))
    rows, failures = [], []
    def acquire(row):
        if dataset == 'llp':
            path, digest, info = llp_video(row)
        else:
            clip = row['clip_id'] if dataset == 'vggss' else row['record']['video_name']
            path, digest, _ = fetch_video(clip, DATA_ROOT/'avqa')
            info = inspect(path)
        result = {**row, 'media_path': str(path), 'media_sha256': digest, **info}
        if dataset == 'avqa':
            normalized = normalize(row['record'], path, 'test', digest, info['duration_s'])
            # New task-specific source role, never a legal Gate training asset.
            result.update(normalized, official_split='evaluation', evaluation_cohort='new_source_disjoint_grounding',
                          has_audio_track=info['has_audio_track'])
        return result
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(acquire, row): row for row in plan}
        for index, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            try: rows.append(future.result())
            except Exception as error:
                failures.append({'clip_id': row.get('clip_id', row.get('record', {}).get('video_name')),
                                 'source_video_id': row['source_video_id'], 'reason': str(error)})
            if index % 16 == 0 or index == len(plan):
                print({'dataset': dataset, 'attempts': index, 'success': len(rows), 'failed': len(failures)}, flush=True)
    key = (lambda r: r['question_id']) if dataset == 'avqa' else (lambda r: r['clip_id'])
    rows.sort(key=key)
    path = root/'datasets'/(dataset+'_manifest.jsonl')
    if path.exists() and read_jsonl(path) != rows:
        raise ValueError('immutable acquired manifest differs')
    write_jsonl(path, rows)
    write_jsonl(root/'datasets'/(dataset+'_acquisition_failures.jsonl'), failures)
    write_json(root/'datasets'/(dataset+'_acquisition.json'), {'planned': len(plan), 'acquired': len(rows),
        'failed': len(failures), 'missing_audio_tracks': sum(not r['has_audio_track'] for r in rows),
        'bytes_of_selected_media': sum(Path(r['media_path']).stat().st_size for r in rows),
        'no_QA_outcome_or_model_output_used_in_acquisition': True, 'manifest_sha256': sha256(path)})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['vggss', 'llp', 'avqa'], required=True)
    p.add_argument('--output', type=Path, default=ROOT)
    args = p.parse_args()
    run(args.dataset, args.output)


if __name__ == '__main__': main()
