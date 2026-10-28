"""Official-training-only AVQA acquisition and source-grouped Gate2 subsets."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import subprocess

from .common import DATA_ROOT, REPO, fingerprint, sha256, validate_splits, write_json, write_jsonl

MEDIA_REPO = 'juyil/AVQA-videos'
MEDIA_REVISION = 'e9538b7abd3dcf018f85c9e431d51f542728c13b'
OFFICIAL_REVISION = 'a5056c1d82849aa0981f50fa4c5a57bd16658c52'


def original_source(clip):
    match = re.fullmatch(r'([A-Za-z0-9_-]{11})_(\d{6})',clip)
    if not match:
        raise ValueError('invalid AVQA original-source/clip identifier')
    return match.group(1),int(match.group(2))


def ranked(values,namespace):
    return sorted(values,key=lambda value:fingerprint([1234,namespace,str(value)]))


def normalize(record,path,split,media_hash,duration):
    source,start=original_source(record['video_name'])
    options=record['multi_choice']
    answer=record['answer']
    if len(options)!=4 or not isinstance(answer,int) or not 0<=answer<4:
        raise ValueError('official AVQA four-choice schema mismatch')
    if record['question_relation'] not in ('View','Sound','Both'):
        raise ValueError('unknown official modality category')
    if duration<2.:
        raise ValueError('clip shorter than first fixed causal arrival')
    return {'question_id':f"avqa:{record['id']}",'videoID':record['video_name'],
        'source_video_id':'youtube:'+source,'source_clip_start_s':start,
        'video_path':str(Path(path).resolve()),'media_sha256':media_hash,'split':split,
        'question':record['question_text'],'options':[f'{letter}. {text}' for letter,text in zip('ABCD',options)],
        'answer':'ABCD'[answer],'question_modality':record['question_relation'],
        'task_type':record['question_type'],'dataset':'avqa_official_split_gate2',
        'official_split':'evaluation' if split=='test' else 'train',
        'query_time_seconds':min(10.,float(duration)),'frame_interval_s':2.,
        'source_duration_seconds':float(duration),'annotation_id':record['id'],
        'annotation_source_commit':OFFICIAL_REVISION,'media_source_revision':MEDIA_REVISION}


def select_records(train,evaluation,available,excluded_sources,counts):
    evaluation_sources={original_source(r['video_name'])[0] for r in evaluation}
    train_sources={original_source(r['video_name'])[0] for r in train}
    overlap=train_sources & evaluation_sources
    groups=defaultdict(list)
    for official,rows in [('train',train),('test',evaluation)]:
        for row in rows:
            source,_=original_source(row['video_name'])
            if source in excluded_sources or row['video_name'] not in available:
                continue
            if official=='train' and source in evaluation_sources:
                continue
            groups[official,source].append(row)
    pool=ranked([source for official,source in groups if official=='train'],'avqa-train-val-source-order')
    boundary=int(len(pool)*.8)
    assigned={'train':set(pool[:boundary]),'val':set(pool[boundary:]),
              'test':{source for official,source in groups if official=='test'}}
    selected=[]
    statistics={}
    for split,sources in assigned.items():
        official='test' if split=='test' else 'train'
        candidates=[]
        for source in sources:
            records=groups[official,source]
            # Rare official categories are retained without examining any answer or prediction.
            record=min(records,key=lambda r:({'View':0,'Sound':1,'Both':2}[r['question_relation']],
                                             fingerprint([1234,'one-qa-per-source',r['id']])))
            candidates.append(record)
        candidates=sorted(candidates,key=lambda r:({'View':0,'Sound':1,'Both':2}[r['question_relation']],
                                                  fingerprint([1234,'source-subset',r['video_name']])))
        if len(candidates)<counts[split]:
            raise ValueError(f'insufficient source-disjoint media for {split}: {len(candidates)}')
        picked=candidates[:counts[split]]
        selected.extend({'record':r,'split':split} for r in picked)
        statistics[split]={'eligible_sources':len(candidates),'chosen':len(picked),
            'modalities':dict(Counter(r['question_relation'] for r in picked))}
    return selected,{'train_eval_source_overlap_removed_from_training':len(overlap),
        'baseline_videomme_sources_excluded':len(excluded_sources),'by_split':statistics,
        'one_QA_per_original_YouTube_source':True,'val_comes_from_official_train_only':True,
        'evaluation_annotations_used_for_training':False,'selection_uses_QA_outcomes':False}


def fetch_video(clip,root):
    import av
    path=root/'videos'/(clip+'.mp4')
    path.parent.mkdir(parents=True,exist_ok=True)
    provenance=path.with_suffix('.source.json')
    url=f'https://huggingface.co/datasets/{MEDIA_REPO}/resolve/{MEDIA_REVISION}/videos/{clip}.mp4?download=true'
    if not path.exists():
        temporary=path.with_suffix('.part')
        headers=path.with_suffix('.headers.tmp')
        command=['curl','--fail','--silent','--show-error','--location','--retry','3','--retry-all-errors',
            '--max-time','120','--dump-header',str(headers),'--output',str(temporary),url]
        result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        if result.returncode:
            raise RuntimeError(f'{clip}: download failed: '+result.stderr[-200:])
        declared=None
        for line in headers.read_text(errors='replace').splitlines():
            if line.lower().startswith('x-linked-etag:'):
                candidate=line.split(':',1)[1].strip().strip('"')
                if re.fullmatch('[0-9a-f]{64}',candidate):
                    declared=candidate
        digest=sha256(temporary)
        if declared is not None and digest!=declared:
            raise ValueError('downloaded media differs from declared LFS hash')
        temporary.replace(path)
        headers.unlink(missing_ok=True)
        write_json(provenance,{'repository':MEDIA_REPO,'revision':MEDIA_REVISION,'source_url':url,
            'sha256':digest,'bytes':path.stat().st_size,'declared_sha256':declared,
            'declared_hash_verified':declared is not None,'access':'anonymous public file endpoint'})
    info=json.loads(provenance.read_text())
    if sha256(path)!=info['sha256'] or info['revision']!=MEDIA_REVISION:
        raise ValueError('existing media/provenance mismatch')
    with av.open(str(path)) as container:
        video=container.streams.video
        audio=container.streams.audio
        if not video:
            raise ValueError('selected file contains no video')
        duration=container.duration/av.time_base if container.duration is not None else float(video[0].duration*video[0].time_base)
        details={'duration_s':duration,'has_audio_track':bool(audio),
                 'audio_sample_rate':audio[0].sample_rate if audio else None}
    return path,info['sha256'],details


def prepare(root,output,metadata,train_count=512,val_count=128,test_count=256):
    root,output,metadata=Path(root),Path(output),Path(metadata)
    source=root/'official_source'
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    if revision!=OFFICIAL_REVISION:
        raise ValueError('official annotation commit changed')
    annotations=source/'data/annotation'
    train=json.loads((annotations/'train_qa.json').read_text())
    evaluation=json.loads((annotations/'val_qa.json').read_text())
    media=json.loads(metadata.read_text())
    if media['sha']!=MEDIA_REVISION or media.get('gated') or media.get('disabled'):
        raise ValueError('expected the pinned publicly available media mirror')
    available={Path(r['rfilename']).stem for r in media['siblings'] if r['rfilename'].startswith('videos/')}
    baseline=json.loads((REPO/'results/adbt_videomme_split.json').read_text())
    excluded={str(x) for subset in ('train','test') for x in baseline[subset]}
    selected,audit=select_records(train,evaluation,available,excluded,
                                  {'train':train_count,'val':val_count,'test':test_count})
    output.mkdir(parents=True,exist_ok=True)
    plan={'official_annotation_commit':revision,'official_annotation_sha256':{
        n:sha256(annotations/n) for n in ('train_qa.json','val_qa.json')},'media_revision':MEDIA_REVISION,
        'media_listing_sha256':sha256(metadata),'listed_media':len(available),'seed':1234,
        'fixed_counts':{'train':train_count,'val':val_count,'test':test_count},
        'selected':selected,'audit':audit,
        'dataset_rights':'original annotation/footage rights retained; public mirror states non-commercial research only',
        'timing':'fixed2s causal arrivals up to actual clip duration capped at10s',
        'ground_truth_use':'train targets only from official train; threshold fit val derived from official train; official val reserved for test'}
    plan_path=output/'selection_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:
        raise ValueError('immutable answer-independent data selection changed')
    write_json(plan_path,plan)
    videos=sorted({x['record']['video_name'] for x in selected})
    acquired={}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures={executor.submit(fetch_video,clip,root):clip for clip in videos}
        for index,future in enumerate(as_completed(futures),1):
            clip=futures[future]
            acquired[clip]=future.result()
            if index%25==0 or index==len(videos):
                print(f'verified AVQA clips {index}/{len(videos)}',flush=True)
    rows=[]
    for item in selected:
        path,digest,details=acquired[item['record']['video_name']]
        row=normalize(item['record'],path,item['split'],digest,details['duration_s'])
        row.update(has_audio_track=details['has_audio_track'])
        rows.append(row)
    validate_splits(rows)
    write_jsonl(output/'qa_manifest.jsonl',rows)
    write_json(output/'split_manifest.json',{'audit':audit,'source_group_audit':validate_splits(rows),
        'annotation_hashes':plan['official_annotation_sha256'],'official_annotation_commit':revision,
        'media_revision':MEDIA_REVISION,'selection_plan_sha256':sha256(plan_path),
        'downloaded_clips':len(acquired),'bytes':sum(p.stat().st_size for p,_,_ in acquired.values()),
        'missing_audio':sum(not r['has_audio_track'] for r in rows),
        'train_source':'official AVQA train_qa.json','val_source':'held-out original sources within official train_qa.json',
        'test_source':'official AVQA val_qa.json, no training use'})
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DATA_ROOT/'avqa')
    parser.add_argument('--output',type=Path,default=DATA_ROOT/'utility_gate/gate2_20261027')
    parser.add_argument('--metadata',type=Path,required=True)
    parser.add_argument('--train',type=int,default=512)
    parser.add_argument('--val',type=int,default=128)
    parser.add_argument('--test',type=int,default=256)
    args=parser.parse_args()
    print({'questions':len(prepare(args.root,args.output,args.metadata,args.train,args.val,args.test))})


if __name__=='__main__':
    main()
