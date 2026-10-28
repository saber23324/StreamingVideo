"""Source-isolated plans, official validation completion and Robust generation."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import os
from pathlib import Path
import random
import shutil

import numpy as np

from utility_gate.common import DATA_ROOT, REPO, fingerprint, read_jsonl, sha256, write_json, write_jsonl
from utility_gate.dual_head.data_audit import CORR_MANIFEST, source
from utility_gate.gate2_data import fetch_video, normalize
from utility_gate.grounding.acquire import inspect, llp_video
from utility_gate.grounding.llp_parser import clip_identity, labels_at, read_intervals
from .protocol import DATA, OLD, ROOT, SEED, read


def ranked(rows,namespace,key=lambda r:r):
    return sorted(rows,key=lambda r:fingerprint([SEED,namespace,key(r)]))


def dense_target(label):
    if label['current_bin_joint']:return 1.,'official_dense_joint_positive',1.
    if label['causal_window_joint']:return None,'unknown_preceding_bin_joint',0.
    if label['subtype']=='neither':return None,'unknown_no_annotated_event',0.
    return 0.,'official_dense_nonjoint_proxy_negative',.75


def prior_sources():
    used=set();evaluation=set()
    for path in (DATA_ROOT/'utility_gate').glob('*/qa_manifest.jsonl'):
        for r in read_jsonl(path):
            sid=source(r['source_video_id']);used.add(sid)
            if r['split']=='test':evaluation.add(sid)
    for r in read_jsonl(CORR_MANIFEST):
        used.add(source(r['source_video_id']));used.add(source(r.get('audio_source_video_id',r['source_video_id'])))
        if r['split']=='test':evaluation.add(source(r['source_video_id']))
    for dataset in ('llp','vggss','avqa','avqa_supplement'):
        for r in read_jsonl(REPO/'artifacts/utility_gate/gate2_grounding_20261027/datasets'/f'{dataset}_manifest.jsonl'):
            sid=source(r['source_video_id']);used.add(sid);evaluation.add(sid)
    owner=read(REPO/'results/adbt_videomme_split.json')
    for split in ('train','test'):
        used.update(map(str,owner[split]));evaluation.update(map(str,owner[split]))
    return used,evaluation


def make_plan(root=ROOT):
    root=Path(root)
    if (root/'datasets/planning_lock.json').exists():raise FileExistsError('plans already frozen')
    origins=read(root/'sources/official_repositories.json')
    folder=Path(origins['YapengTian/AVVP-ECCV20']['path'])/'data'
    used,old_evaluation=prior_sources()
    oldutility=[r for r in read_jsonl(OLD/'training_assets/utility_train_val.jsonl') if r['supervision_scope']=='measured_window']
    roles={source(r['source_video_id']):r['dataset_role'] for r in oldutility}
    evalqa=read(DATA_ROOT/'avqa/official_source/data/annotation/val_qa.json')
    all_evalqa_sources={r['video_name'][:11] for r in evalqa}
    full=list(csv.DictReader((folder/'AVVP_dataset_full.csv').open(),delimiter='\t'))
    full_count=Counter()
    for r in full:
        try:full_count[clip_identity(r['filename'])[0]]+=1
        except ValueError:pass
    mirror=read(DATA_ROOT/'llp_avvp/mirror_UnFaZeD07.json')
    media={Path(r['rfilename']).stem:r['rfilename'] for r in mirror['siblings'] if r['rfilename'].startswith('videos/')}
    dense_rejections=[]
    audio=read_intervals(folder/'AVVP_eval_audio.csv',False,dense_rejections)
    visual=read_intervals(folder/'AVVP_eval_visual.csv',False,dense_rejections)
    bad={r['filename'] for r in dense_rejections}
    split_rows={s:list(csv.DictReader((folder/f'AVVP_{s}_pd.csv').open(),delimiter='\t')) for s in ('val','test')}
    test_sources={r['filename'][:11] for r in split_rows['test']}
    llp={};rejected=[]
    for split,records in split_rows.items():
        pool=[]
        for r in records:
            try:sid,start,_=clip_identity(r['filename'])
            except ValueError:continue
            reasons=[]
            if r['filename'] in bad:reasons.append('malformed_dense_annotation')
            if full_count[sid]!=1:reasons.append('ambiguous_mirror_source_offset')
            if sid not in media:reasons.append('no_pinned_exact_media')
            if split=='val' and sid in old_evaluation|all_evalqa_sources|test_sources:reasons.append('evaluation_source_reserved')
            if split=='test' and sid in used:reasons.append('previously_used_source')
            if reasons:
                rejected.append({'clip_id':r['filename'],'official_split':split,'reasons':reasons});continue
            pool.append({'clip_id':r['filename'],'source_video_id':sid,'clip_offset_s':start,
                'canonical_clip_id':sid+f'_{start:06d}','official_split':split,
                'mirror_path':media[sid],'media_path':None})
        pool=ranked(pool,'LLP-'+split,lambda r:r['clip_id'])
        if split=='val':
            chosen=pool[:480]
            for row in chosen:
                sid=row['source_video_id']
                role=roles.get(sid,'train' if int(fingerprint([SEED,'global-role',sid])[:8],16)%5 else 'val')
                roles[sid]=role;row['split']=role
        else:
            chosen=[r for r in pool if r['source_video_id'] not in roles][:128]
            for row in chosen:row['split']='test'
        llp[split]=chosen
    write_jsonl(root/'datasets/llp_development_plan.jsonl',llp['val'])
    write_jsonl(root/'datasets/llp_confirmation_plan.jsonl',llp['test'])
    # Dense development annotations only. Confirmation labels are loaded by
    # the evaluator after the preprediction lock, never by this trainer.
    rows=[]
    for r in llp['val']:
        for t in range(1,11):
            label=labels_at(audio.get(r['clip_id'],[]),visual.get(r['clip_id'],[]),t)
            target,strength,weight=dense_target(label)
            rows.append({**r,'time_s':t,**label,'corr_target':target,'label_strength':strength,'label_weight':weight})
    write_jsonl(root/'labels/llp_development_annotations.jsonl',rows)
    avslookup={r['pair_id']:r for r in read_jsonl(CORR_MANIFEST)}
    avs=[]
    for r in read_jsonl(OLD/'training_assets/correspondence_train_val.jsonl'):
        if r['official_split']!='train' or r['corr_target']!=1:continue
        original=avslookup[r['row_id']];sid=source(r['source_video_id'])
        if sid in test_sources|old_evaluation|all_evalqa_sources:continue
        role=roles.get(sid,'train' if int(fingerprint([SEED,'global-role',sid])[:8],16)%5 else 'val')
        roles[sid]=role
        avs.append({**original,'split':role,'official_split':'train','feature_cache_path':r['feature_cache_path'],
            'feature_cache_key':r['feature_cache_key'],'source_clip_start_s':r['source_clip_start_s']})
    write_jsonl(root/'datasets/avs_original_training_plan.jsonl',avs)
    # Newly acquired QA development and confirmation: no model scores or
    # answer correctness are consulted in source/question selection.
    trainqa=read(DATA_ROOT/'avqa/official_source/data/annotation/train_qa.json')
    listing=read(DATA_ROOT/'avqa/acquisition_20261027/media_repository.json')
    available={Path(x['rfilename']).stem for x in listing['siblings'] if x['rfilename'].startswith('videos/')}
    llp_reserved={r['filename'][:11] for records in split_rows.values() for r in records}
    vgg={r['file'][:11] for r in read(DATA_ROOT/'vggss/annotations/raw/vggss.json')}
    def one_per_source(records,excluded):
        grouped=defaultdict(list)
        for r in records:
            if r['video_name'] in available and r['video_name'][:11] not in excluded:grouped[r['video_name'][:11]].append(r)
        prefer=lambda r:({'View':0,'Sound':1,'Both':2}[r['question_relation']],fingerprint([SEED,'question',r['id']]))
        return sorted([min(group, key=prefer) for group in grouped.values()],key=prefer)
    development=one_per_source(trainqa,used|set(roles)|llp_reserved|all_evalqa_sources)
    if len(development)<96:raise ValueError('fewer than96 fresh official train sources')
    qadev=[]
    for i,r in enumerate(development[:96]):
        split='train' if i<64 else 'val'
        # A seeded order first, rather than placing all rare categories in train.
        qadev.append({'record':r,'source_video_id':r['video_name'][:11],'split':split})
    qadev=ranked(qadev,'QA-development-assignment',lambda r:r['source_video_id'])
    for i,r in enumerate(qadev):r['split']='train' if i<64 else 'val';roles[r['source_video_id']]=r['split']
    confirmation=one_per_source(evalqa,used|set(roles)|llp_reserved|vgg)[:128]
    if len(confirmation)<128:raise ValueError('fewer than128 unused official evaluation sources')
    qatest=[{'record':r,'source_video_id':r['video_name'][:11],'split':'test'} for r in confirmation]
    write_jsonl(root/'datasets/qa_development_plan.jsonl',qadev)
    write_jsonl(root/'datasets/qa_confirmation_plan.jsonl',qatest)
    write_json(root/'audit/source_roles.json',roles)
    write_jsonl(root/'audit/LLP_exclusions.jsonl',rejected)
    write_jsonl(root/'audit/dense_annotation_rejections.jsonl',dense_rejections)
    write_json(root/'audit/previously_used_sources.json',sorted(used))
    summaries={'LLP_official_val':len(split_rows['val']),'LLP_official_test':len(split_rows['test']),
        'LLP_development':dict(Counter(r['split'] for r in llp['val'])),'LLP_confirmation_planned':len(llp['test']),
        'LLP_development_labels':dict(Counter(r['label_strength'] for r in rows)),
        'AVS_original_train':dict(Counter(r['split'] for r in avs)),
        'QA_new_development':dict(Counter(r['split'] for r in qadev)),
        'QA_confirmation_planned':len(qatest),'QA_confirmation_public_categories':dict(Counter(r['record']['question_relation'] for r in qatest)),
        'previous_sources_excluded_from_confirmation':len(used),'selection_uses_answers_or_predictions':False}
    write_json(root/'datasets/planning_lock.json',{'summary':summaries,
        'plans_sha256':{p.name:sha256(p) for p in (root/'datasets').glob('*plan.jsonl')},
        'test_labels_used_to_fit':False,'official_Test_never_train_or_calibration':True})
    print(summaries,flush=True)


def acquire(root=ROOT):
    root=Path(root);manifests={}
    for name in ('llp_development','llp_confirmation','qa_development','qa_confirmation'):
        plan=read_jsonl(root/'datasets'/f'{name}_plan.jsonl');rows=[];failures=[]
        def get(row):
            if name.startswith('llp'):
                path,digest,info=llp_video(row)
                return {**row,'media_path':str(path),'media_sha256':digest,**info}
            path,digest,info=fetch_video(row['record']['video_name'],DATA_ROOT/'avqa')
            return {**normalize(row['record'],path,row['split'],digest,info['duration_s']),
                'has_audio_track':info['has_audio_track']}
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures={executor.submit(get,r):r for r in plan}
            for i,future in enumerate(as_completed(futures),1):
                try:rows.append(future.result())
                except Exception as error:failures.append({'source_video_id':futures[future]['source_video_id'],'reason':str(error)})
                if i%32==0 or i==len(plan):print({'cohort':name,'attempted':i,'acquired':len(rows),'failed':len(failures)},flush=True)
        rows.sort(key=lambda r:r['source_video_id'])
        write_jsonl(root/'datasets'/f'{name}_manifest.jsonl',rows)
        write_jsonl(root/'datasets'/f'{name}_failures.jsonl',failures);manifests[name]=rows
    # Source and byte aliases must not cross any local role, or reuse a prior
    # source in confirmation, including AVS audio donors in the next step.
    assigned={};used=set(read(root/'audit/previously_used_sources.json'))
    for rows in manifests.values():
        for r in rows:
            sid=source(r['source_video_id']);role=r['split']
            if role=='test' and sid in used:raise AssertionError('previous source entered confirmation')
            for key in [('source',sid),('bytes',r['media_sha256'])]:
                if key in assigned and assigned[key]!=role:raise AssertionError('cross-role source/media alias')
                assigned[key]=role
    write_json(root/'datasets/acquisition_summary.json',{name:{'acquired':len(rows),'sources':len({source(r['source_video_id']) for r in rows}),
        'by_split':dict(Counter(r['split'] for r in rows)),'missing_audio':sum(not r['has_audio_track'] for r in rows)} for name,rows in manifests.items()})


def generate_robust(root=ROOT):
    """Execute pinned official S4 donor-selection body on train-only staging.

    Only the export/feature callback and valid-PCM noise helper are adapted;
    raw donor MP4, not an already-resampled export, feeds causal BEATs later.
    """
    from pydub import AudioSegment
    root=Path(root);rows=read_jsonl(root/'datasets/avs_original_training_plan.jsonl')
    original=Path(read(root/'sources/official_repositories.json')['jiali-home/AVSBench-Robust']['path'])/'dataset_generation/audio_processing.py'
    tree=ast.parse(original.read_text())
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('get_semantic_category','process_s4_split')]
    if len(selected)!=2:raise ValueError('official generator definitions changed')
    inputs=DATA/'robust/inputs';outputs=DATA/'robust/generated';lookup=defaultdict(list)
    for row in rows:
        wav=Path(row['archived_audio_path']);target=inputs/row['split']/row['category']/wav.name
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():os.link(wav,target)
        audio=AudioSegment.from_wav(wav)
        lookup[hashlib.sha256(audio.raw_data).hexdigest()].append(row)
    generated=[]
    def save(audio,path,feature_path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);audio.export(path,format='wav')
        if 'misaligned' not in str(path):return
        role,category,filename=path.parts[-3:];recipient=next(r for r in rows if r['split']==role and r['source_video_id']==Path(filename).stem)
        donors=[r for r in lookup[hashlib.sha256(audio.raw_data).hexdigest()] if r['split']==role]
        if len(donors)!=1:raise ValueError('ambiguous donor identity in official synthetic output')
        donor=donors[0]
        if recipient['source_video_id']==donor['source_video_id'] or recipient['category']==donor['category']:raise AssertionError('nonmismatched donor')
        generated.append({'row_id':'robust:'+recipient['pair_id'],'split':role,'official_split':'train',
            'source_video_id':recipient['source_video_id'],'audio_source_video_id':donor['source_video_id'],
            'category':recipient['category'],'donor_category':donor['category'],'frame_timestamp_s':recipient['frame_timestamp_s'],
            'original_feature_cache_path':recipient['feature_cache_path'],'raw_donor_media_path':donor['audio_path'],
            'visual_media_sha256':recipient['source_media_sha256'],'audio_media_sha256':donor['source_media_sha256'],
            'generated_audio_path':str(path),'generated_audio_sha256':sha256(path),
            'corr_target':0.,'label_strength':'synthetic_offscreen_proxy_negative','label_weight':.5,
            'generator_sha256':sha256(original),'seed':42,
            'BEATs_input':'raw donor native-rate MP4 causal prefix, never whole-track-resampled WAV'})
    rng=np.random.default_rng(42)
    def noise(duration_ms=5000,sample_rate=16000):
        pcm=rng.integers(-12000,12000,int(sample_rate*duration_ms/1000),dtype=np.int16)
        return AudioSegment(pcm.tobytes(),frame_rate=sample_rate,sample_width=2,channels=1)
    namespace={'os':os,'random':random.Random(42),'AudioSegment':AudioSegment,
               'save_audio_and_features':save,'generate_noise_audio':noise}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(original),'exec'),namespace)
    for role in ('train','val'):
        namespace['process_s4_split'](str(inputs),role,str(outputs/'misaligned'),str(outputs/'unused_logmel'),str(outputs/'noise'),str(outputs/'unused_noise_logmel'))
    write_jsonl(root/'datasets/avs_robust_generated.jsonl',generated)
    write_json(root/'datasets/avs_robust_generation.json',{'generated_mismatch':dict(Counter(r['split'] for r in generated)),
        'official_generator_sha256':sha256(original),'official_functions':['get_semantic_category','process_s4_split'],
        'adaptations':['only original official train sources staged into local train/calibration','BEATs replaces unused VGGish/logmel export',
            'noise helper uses valid PCM16; official float64-bytes/sample_width2 mismatch not replicated','causal features decode raw donor instead of resampled WAV'],
        'noise_is_not_a_semantic_negative_training_target':True,'labels_are_synthetic_proxy_not_true_invisibility':True,
        'original_access':'previously acquired public research media with original AVSBench rights retained; no new gated access or permission claim',
        'old_media_and_checkpoints_overwritten':False})
    print({'generated_mismatches':len(generated)},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['plan','acquire','robust']);args=p.parse_args()
    {'plan':make_plan,'acquire':acquire,'robust':generate_robust}[args.stage]()
