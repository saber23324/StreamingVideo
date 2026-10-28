"""Extract all newly independent rare AVQA relations available in public MTEB.

Reads only the required Parquet row groups. Uses original media bytes, never
precomputed embeddings; official question/options must match byte-for-byte.
"""
from collections import Counter, defaultdict
import io
from pathlib import Path

import av
import pyarrow.parquet as pq

from utility_gate.common import DATA_ROOT, read_jsonl, sha256, write_json, write_jsonl
from utility_gate.gate2_data import normalize
from .acquire import inspect
from .dataset_audit import ROOT, all_exclusions, official_vggss, read
from .remote_zip import RemoteConcat


def run():
    data = DATA_ROOT/'avqa'
    index = read(data/'mteb_video_index.json')
    official = read(data/'official_source/data/annotation/val_qa.json')
    by_clip = defaultdict(list)
    for r in official: by_clip[r['video_name']].append(r)
    _, _, seen, baseline = all_exclusions()
    exclude = seen|baseline|{r['file'][:11] for r in official_vggss()}|{
        r['source_video_id'] for r in read_jsonl(ROOT/'datasets/llp_plan.jsonl')}
    selected = []
    for row in index['rows']:
        clip = Path(row['video_id']).stem
        match = [r for r in by_clip[clip] if r['question_text'] == row['question'] and r['multi_choice'] == row['candidates']]
        if len(match) != 1: raise ValueError('MTEB row does not match official QA identity')
        if match[0]['question_relation'] != 'Both' and clip[:11] not in exclude:
            selected.append({'record': match[0], 'shard': row['shard'], 'row_index': row['row'], 'video_id': row['video_id']})
    path = ROOT/'datasets/avqa_supplement_plan.jsonl'
    if path.exists() and read_jsonl(path) != selected: raise ValueError('supplement plan is immutable')
    write_jsonl(path, selected)
    trees = {r['path']: r for r in read(data/'mteb_tree.json')}
    target = DATA_ROOT/'avqa/grounding_mteb_media'; target.mkdir(exist_ok=True)
    output, failures = [], []
    for shard_name in sorted({r['shard'] for r in selected}):
        descriptor = trees[shard_name]
        url = 'https://huggingface.co/datasets/mteb/AVQA_val/resolve/'+index['revision']+'/'+shard_name+'?download=true'
        remote = RemoteConcat([(url, descriptor['size'])], data/'mteb_range_cache')
        parquet = pq.ParquetFile(remote)
        boundaries = [0]
        for g in range(parquet.num_row_groups): boundaries.append(boundaries[-1]+parquet.metadata.row_group(g).num_rows)
        table_cache = {}
        for item in [r for r in selected if r['shard'] == shard_name]:
            try:
                i = item['row_index']; group = next(g for g in range(len(boundaries)-1) if boundaries[g] <= i < boundaries[g+1])
                if group not in table_cache:
                    table_cache[group] = parquet.read_row_group(group, use_threads=False).to_pylist()
                row = table_cache[group][i-boundaries[group]]
                if row['video_id'] != item['video_id']: raise ValueError('row group identity changed')
                clip = item['record']['video_name']; path = target/(clip+'.mp4')
                video_bytes = row['video']['bytes']; audio_bytes = row['audio']['bytes']
                if not video_bytes or not audio_bytes: raise ValueError('media bytes absent')
                path.write_bytes(video_bytes)
                audio_path = target/(clip+'.audio')
                audio_path.write_bytes(audio_bytes)
                details = inspect(path)
                with av.open(io.BytesIO(audio_bytes)) as container:
                    audio_duration = container.duration/av.time_base
                if abs(audio_duration-details['duration_s']) > .25: raise ValueError('separate AV media duration mismatch')
                if details['duration_s'] < 8: raise ValueError('supplement not a complete nominal10s clip')
                record = normalize(item['record'], path, 'test', sha256(path), details['duration_s'])
                record.update(media_path=str(path), **details, external_audio_path=str(audio_path),
                    external_audio_sha256=sha256(audio_path), source_mirror='mteb/AVQA_val', source_revision=index['revision'],
                    parquet_shard=shard_name, parquet_row=i, evaluation_cohort='new_source_disjoint_grounding',
                    media_source_revision=index['revision'], native_encoder_inputs='original video bytes + paired audio bytes, not cached features')
                write_json(path.with_suffix('.source.json'), record)
                output.append(record)
                print({'extracted_new_rare': clip, 'relation': record['question_modality'], 'video_audio_track': details['has_audio_track']}, flush=True)
            except Exception as error:
                failures.append({'video_id': item['video_id'], 'reason': str(error)})
        write_json(target/(Path(shard_name).stem+'.ranges.json'), remote.requests)
    write_jsonl(ROOT/'datasets/avqa_supplement_manifest.jsonl', output)
    write_jsonl(ROOT/'datasets/avqa_supplement_failures.jsonl', failures)
    write_json(ROOT/'datasets/avqa_supplement_acquisition.json', {'selected': len(selected), 'acquired': len(output),
        'failures': failures, 'by_relation': dict(Counter(r['question_modality'] for r in output)),
        'selection_uses_predictions': False, 'source': 'public921-row MTEB subset; exact official QA/options matching'})


if __name__ == '__main__': run()
