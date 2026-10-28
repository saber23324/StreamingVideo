"""Frozen native causal caches and measured QA-window labels; no test fitting."""
from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path

import numpy as np
import torch

from utility_gate.common import append_jsonl, fingerprint, read_jsonl, seed_all, sha256, write_json, write_jsonl
from utility_gate.features import feature_payload, take_features
from utility_gate.gate2_interfaces import assemble_anchor_features, features_from_cache
from utility_gate.gate2_runtime import qualified_latents
from utility_gate.grounding.native import encode_times
from relevance_gate.media import frame_at_or_before
from utility_gate.runtime import score_answer
from utility_gate.strict_runtime import StrictStreamingRuntime
from .protocol import DATA, OLD, ROOT, SEED, read


def save_tensors(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():return
    temporary=path.with_suffix(f'.part.{os.getpid()}')
    with temporary.open('wb') as stream:torch.save(value,stream)
    try:os.link(temporary,path)
    except FileExistsError:pass
    finally:temporary.unlink(missing_ok=True)


@torch.inference_mode()
def llp_features(runtime,cohort='development',root=ROOT,shard=None,shards=1):
    root=Path(root)
    if cohort=='confirmation' and not (root/'pretest_lock.json').exists():raise ValueError('confirmation requires a preprediction lock')
    records=read_jsonl(root/'datasets'/f'llp_{cohort}_manifest.jsonl')
    if shard is not None:records=[r for j,r in enumerate(records) if j%shards==shard]
    results=[];failures=[]
    for i,row in enumerate(records,1):
        try:
            key=fingerprint([row['media_sha256'],'LLP-strict-half-open-bin-1s-and2s-v2',runtime.metadata['encoder']])
            path=DATA/'features/llp_strict_bin'/(key+'.pt')
            if not path.exists():
                if sha256(row['media_path'])!=row['media_sha256']:raise ValueError('media hash changed')
                times=[t for t in range(1,11) if t<=row['duration_s']+1e-6]
                legacy_key=fingerprint([row['media_sha256'],'LLP-native-1s-and2s-v1',runtime.metadata['encoder']])
                legacy=DATA/'features/llp'/(legacy_key+'.pt')
                if legacy.exists():
                    old=torch.load(legacy,map_location='cpu',weights_only=True)['grids']['1s']
                    visual_mean=old['features']['visual'][:,:3584].to(runtime.device).clone()
                    audio_mean=old['features']['audio'][:,:768].to(runtime.device)
                    valid=old['features']['valid_audio'].to(runtime.device)
                    quality=old['quality'];pts=list(old['frame_pts_s'])
                    replace_indices=[j for j,t in enumerate(times) if not t-1<=pts[j]<t]
                    frames=[frame_at_or_before(row['media_path'],times[j]-1e-6) for j in replace_indices]
                    if frames:
                        updated=runtime.extractor.encode_visual(runtime.model,runtime.processor,np.stack([f[0] for f in frames]),1).float().mean(1)
                        visual_mean[replace_indices]=updated
                        for j,(_,point) in zip(replace_indices,frames):pts[j]=point
                else:
                    frames=[frame_at_or_before(row['media_path'],t-1e-6) for t in times]
                    visual_mean=runtime.extractor.encode_visual(runtime.model,runtime.processor,np.stack([f[0] for f in frames]),1).float().mean(1)
                    windows,valid,quality,_=runtime.causal_windows(row['media_path'],times)
                    audio_mean=runtime.audio_encoder.encode_windows(windows,runtime.device).to(runtime.device).float().mean(1)
                    valid=valid.to(runtime.device);pts=[f[1] for f in frames];replace_indices=[]
                if any(not t-1<=point<t for t,point in zip(times,pts)):raise ValueError('no visual frame strictly inside annotated bin')
                v=visual_mean[None];a=audio_mean[None];valid=valid[None]
                stamp=torch.tensor([times],dtype=torch.float32,device=runtime.device)
                grids={}
                for name,step in [('1s',1),('2s',2)]:
                    index=[j for j,t in enumerate(times) if t%step==0]
                    values=assemble_anchor_features(v[:,index],a[:,index],valid[:,index],stamp[:,index])
                    grids[name]={'features':feature_payload(values),'times':[times[j] for j in index],
                        'quality':[quality[j] for j in index],
                        'frame_pts_s':[pts[j] for j in index]}
                save_tensors(path,{'key':key,'source_video_id':row['source_video_id'],'media_sha256':row['media_sha256'],
                    'official_split':row['official_split'],'split':row['split'],'grids':grids,
                    'raw_causal_decode_version':3,'labels_used_in_encoding':False,
                    'visual_frame_strictly_inside_annotation_bin':True,'audio_decision_endpoint_unchanged':True,
                    'legacy_boundary_frames_corrected':len(replace_indices)})
            cached=torch.load(path,map_location='cpu',weights_only=True)
            if cached['key']!=key or cached['split']!=row['split']:raise ValueError('stale LLP cache role')
            results.append({**row,'features_path':str(path),'features_sha256':sha256(path)})
        except Exception as error:failures.append({'clip_id':row['clip_id'],'reason':str(error)})
        if i%16==0 or i==len(records):print({'LLP':cohort,'attempted':i,'cached':len(results),'failed':len(failures)},flush=True)
    suffix='' if shard is None else f'.prewarm{shard}'
    write_jsonl(root/'features'/f'llp_{cohort}{suffix}.jsonl',results)
    write_jsonl(root/'features'/f'llp_{cohort}{suffix}_failures.jsonl',failures)
    return results


@torch.inference_mode()
def avs_features(runtime,root=ROOT):
    root=Path(root);rows=[]
    for kind,path in [('AVS_original',root/'datasets/avs_original_training_plan.jsonl'),
                      ('AVS_synthetic',root/'datasets/avs_robust_generated.jsonl')]:
        for row in read_jsonl(path):
            original=torch.load(row['feature_cache_path'] if kind=='AVS_original' else row['original_feature_cache_path'],map_location='cpu',weights_only=True)
            stamp=float(row['frame_timestamp_s']);v=original['visual'].float().mean(0)[None,None].to(runtime.device)
            if kind=='AVS_original':
                a=original['audio'].float().mean(0)[None,None].to(runtime.device)
                available=bool(original['valid_aux']);reason=original['quality_reason']
            else:
                windows,valid,reasons,_=runtime.causal_windows(row['raw_donor_media_path'],[stamp])
                a=runtime.audio_encoder.encode_windows(windows,runtime.device).to(runtime.device).float().mean(1)[None]
                available=bool(valid[0]);reason=reasons[0]
            feature=assemble_anchor_features(v,a,torch.tensor([[available]],device=runtime.device),torch.tensor([[stamp]],device=runtime.device))
            key=fingerprint([kind,row.get('pair_id',row.get('row_id')),row.get('audio_media_sha256'),runtime.metadata['encoder']])
            cache=DATA/'features/avs'/(key+'.pt')
            save_tensors(cache,{'key':key,'split':row['split'],'source_video_id':row['source_video_id'],
                'grids':{'single':{'features':feature_payload(feature)}},'raw_causal_decode_version':3})
            rows.append({'row_id':row.get('row_id',row.get('pair_id')),'kind':kind,'split':row['split'],'official_split':'train',
                'source_video_id':row['source_video_id'],'audio_source_video_id':row.get('audio_source_video_id',row['source_video_id']),
                'visual_media_sha256':row.get('visual_media_sha256',row.get('source_media_sha256')),
                'audio_media_sha256':row['audio_media_sha256'],'features_path':str(cache),'features_sha256':sha256(cache),
                'grid':'single','position':0,'time_s':stamp,'quality_valid':available,'quality_reason':reason,
                'corr_target':1. if kind=='AVS_original' else 0.,
                'label_strength':'official_AVS_visible_mask_positive' if kind=='AVS_original' else row['label_strength'],
                'corr_weight':1. if kind=='AVS_original' else .5,'utility_target':None})
        print({'AVS_condition':kind,'rows_so_far':len(rows)},flush=True)
    write_jsonl(root/'labels/avs_training.jsonl',rows)
    return rows


def pooled_qa(payload):
    feature=features_from_cache(payload)
    return {'grids':{'2s':{'features':feature_payload(feature)}},'raw_causal_decode_version':3,
        'split':payload['split'],'source_video_id':payload['source_video_id']}


def import_old_utility(root=ROOT):
    root=Path(root);rows=[];paths={}
    for row in read_jsonl(OLD/'training_assets/utility_train_val.jsonl'):
        if row['supervision_scope']!='measured_window':continue
        if row['official_split']!='train':raise ValueError('old utility eval row')
        original=row['feature_cache_path']
        if original not in paths:
            payload=torch.load(original,map_location='cpu',weights_only=True)
            path=DATA/'features/old_utility'/(row['feature_cache_key']+'.pt')
            save_tensors(path,pooled_qa(payload));paths[original]=str(path)
        rows.append({'row_id':'old:'+row['row_id'],'kind':'Utility','split':row['dataset_role'],'official_split':'train',
            'source_video_id':row['source_video_id'].removeprefix('youtube:'),'media_sha256':row['media_sha256'],
            'features_path':paths[original],'grid':'2s','position':row['window_index'],
            'time_s':row['window_timestamp_s'],'quality_valid':row['eligible_audio'],
            'corr_target':None,'corr_weight':0.,'utility_target':row['nll_utility'],'accuracy_utility':row['accuracy_utility'],
            'label_strength':'measured_old_QA_window','QA_source':'official AVQA train only','label_weight':1.})
    write_jsonl(root/'labels/old_utility.jsonl',rows)
    return rows


@torch.inference_mode()
def qa_development(runtime,root=ROOT):
    root=Path(root);out=root/'labels/new_utility.jsonl';paired=root/'labels/new_QA_development_pairs.jsonl'
    prior=read_jsonl(out) if out.exists() else [];done_rows={r['row_id'] for r in prior}
    done={r['source_video_id'] for r in read_jsonl(paired)} if paired.exists() else set()
    for i,row in enumerate(read_jsonl(root/'datasets/qa_development_manifest.jsonl'),1):
        sid=row['source_video_id'].removeprefix('youtube:')
        if sid in done:continue
        if row['official_split']!='train' or row['split'] not in ('train','val'):raise ValueError('test QA cannot generate training utility')
        try:
            key=fingerprint([row['media_sha256'],'new-QA-strict-v1',runtime.metadata])
            path=DATA/'features/qa_development'/(key+'.pt')
            if path.exists():payload=torch.load(path,map_location='cpu',weights_only=True)
            else:
                payload=runtime.encode(row);payload.update(split=row['split'],source_video_id=row['source_video_id'],cache_key=key)
                save_tensors(path,payload)
            audio,_=qualified_latents(runtime,payload);audio=audio[0]
            vision=payload['vision_latents'].to(runtime.device);prompt=runtime.prompt(row)
            baseline=score_answer(runtime.generate(prompt,audio),row['answer'],row['dataset'])
            reference=score_answer(runtime.generate(prompt,vision),row['answer'],row['dataset'])
            pooled=DATA/'features/qa_pooled'/(key+'.pt');save_tensors(pooled,pooled_qa(payload))
            for j,t in enumerate(payload['timestamps'].tolist()):
                if f"new:{row['question_id']}:{j}" in done_rows:continue
                counterfactual=audio.clone();counterfactual[j]=vision[j]
                identical=torch.equal(counterfactual,audio)
                scored=baseline if identical else score_answer(runtime.generate(prompt,counterfactual),row['answer'],row['dataset'])
                append_jsonl(out,{'row_id':f"new:{row['question_id']}:{j}",'kind':'Utility','split':row['split'],'official_split':'train',
                    'source_video_id':sid,'media_sha256':row['media_sha256'],'features_path':str(pooled),'grid':'2s','position':j,
                    'time_s':t,'quality_valid':bool(payload['audio_valid'][j]),'quality_reason':payload['audio_quality'][j],
                    'corr_target':None,'corr_weight':0.,'utility_target':scored['gold_nll']-baseline['gold_nll'],
                    'accuracy_utility':int(baseline['correct'])-int(scored['correct']),
                    'label_strength':'measured_new_QA_window','label_weight':1.,'actual_counterfactual_generation':not identical,
                    'scope':'single window switched to stage1, all other windows remain qualified audio'})
            append_jsonl(paired,{'source_video_id':sid,'question_id':row['question_id'],'split':row['split'],
                'cache_path':str(path),'cache_sha256':sha256(path),'pooled_features_path':str(pooled),
                'qualified_audio':baseline,'vision_only':reference,'source_record':row,
                'all_generations_actual_except_explicit_identical_bank_window_reuse':True})
        except Exception as error:
            append_jsonl(root/'features/qa_development_failures.jsonl',{'question_id':row['question_id'],'reason':str(error)})
            raise
        if i%8==0:print({'new_QA_development_completed':i,'source':sid},flush=True)


def assemble(root=ROOT):
    root=Path(root)
    annotations={(r['clip_id'],r['time_s']):r for r in read_jsonl(root/'labels/llp_development_annotations.jsonl')}
    rows=[]
    for cached in read_jsonl(root/'features/llp_development.jsonl'):
        payload=torch.load(cached['features_path'],map_location='cpu',weights_only=True);grid=payload['grids']['2s']
        for i,t in enumerate(grid['times']):
            label=annotations[cached['clip_id'],t]
            rows.append({'row_id':f"{cached['clip_id']}:{t}",'kind':'LLP','split':cached['split'],'official_split':'val',
                'source_video_id':cached['source_video_id'],'media_sha256':cached['media_sha256'],
                'features_path':cached['features_path'],'grid':'2s','position':i,'time_s':t,
                'quality_valid':bool(grid['features']['valid_audio'][i]),'quality_reason':grid['quality'][i],
                'corr_target':label['corr_target'],'label_strength':label['label_strength'],'corr_weight':label['label_weight'],
                'utility_target':None,'current_bin_joint':label['current_bin_joint'],'subtype':label['subtype']})
    for name in ('avs_training','old_utility','new_utility'):rows+=read_jsonl(root/'labels'/f'{name}.jsonl')
    from .training import guard_rows
    for split in ('train','val'):guard_rows([r for r in rows if r['split']==split],split,root)
    write_jsonl(root/'labels/training_manifest.jsonl',rows)
    write_json(root/'labels/supervision_summary.json',{
        split:{kind:{'rows':len(group),'sources':len({r['source_video_id'] for r in group}),
            'valid':sum(r['quality_valid'] for r in group),'corr_labels':dict(Counter('unknown' if r['corr_target'] is None else str(r['corr_target']) for r in group)),
            'strength':dict(Counter(r['label_strength'] for r in group)),
            'accuracy_delta_eligible':dict(Counter(str(r.get('accuracy_utility')) for r in group if r['quality_valid'] and r['utility_target'] is not None))}
            for kind in ('LLP','AVS_original','AVS_synthetic','Utility') if (group:=[r for r in rows if r['split']==split and r['kind']==kind])}
        for split in ('train','val')})


def development(root=ROOT):
    root=Path(root);seed_all(SEED);runtime=StrictStreamingRuntime()
    write_json(root/'features/runtime.json',runtime.metadata)
    write_json(root/'audit/new_run_legacy_GPU_fixture.json',runtime.regression_fixture(
        Path('/home/yxd/audiogate/artifacts/relevance_gate/m0-gate-disabled-final-20261022')))
    llp_features(runtime,'development',root);avs_features(runtime,root);import_old_utility(root);qa_development(runtime,root);assemble(root)
    print({'development_features_and_labels':'complete'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--llp-prewarm-shard',type=int);parser.add_argument('--shards',type=int,default=3)
    parser.add_argument('--cohort',choices=['development','confirmation'],default='development')
    args=parser.parse_args()
    if args.llp_prewarm_shard is None:development()
    else:
        seed_all(SEED);llp_features(StrictStreamingRuntime(),args.cohort,ROOT,args.llp_prewarm_shard,args.shards)
