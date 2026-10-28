"""One locked evaluation on new sources; never call training or calibration."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from utility_gate.common import DATA_ROOT, REPO, append_jsonl, read_jsonl, seed_all, sha256, write_json, write_jsonl
from utility_gate.dual_head.diagnostics import paired_source_interval, spearman_source_interval
from utility_gate.dual_head.model import load_dual_head
from utility_gate.features import load_feature_payload, take_features
from utility_gate.gate2_interfaces import features_from_cache, load_gate2
from utility_gate.gate2_runtime import qualified_latents
from utility_gate.grounding.llp_parser import labels_at, read_intervals
from utility_gate.grounding.stats import binary_cluster_ci, mean_ci
from utility_gate.runtime import score_answer
from utility_gate.strict_runtime import StrictStreamingRuntime
from .data import dense_target
from .features import llp_features, save_tensors
from .protocol import DATA, ROOT, SEED, read
from .training import MODES


def apply_rule(r,u,available,rule):
    r,u,available=np.asarray(r),np.asarray(u),np.asarray(available,bool)
    if r.shape!=u.shape or r.shape!=available.shape:raise ValueError('same-sized per-window head scores required')
    if rule.get('fit_role')!='val':raise ValueError('explicit validation-only frozen policy required')
    if not np.isfinite(r).all() or not np.isfinite(u).all():raise ValueError('nonfinite policy inputs')
    if rule['mode']=='corr':selected=r>float(rule['corr_threshold'])
    elif rule['mode']=='utility':selected=u>float(rule['utility_threshold'])
    elif rule['mode']=='dual':selected=(r>float(rule['selection_corr_threshold']))&(u>float(rule['selection_utility_threshold']))
    else:raise ValueError('unknown explicitly proxy-trained routing policy')
    return selected&available


@torch.inference_mode()
def predict(model,features):
    # One causal observation per call also prevents a future batch-size change
    # from perturbing the numerical kernel used for a past decision.
    output=[model.forward_features(take_features(features,slice(i,i+1)).to(next(model.parameters()).device)) for i in range(len(features.visual))]
    return np.asarray([float(o.correspondence_logit[0,0]) for o in output]),np.asarray([float(o.utility_delta[0,0]) for o in output])


def verify_lock(root=ROOT):
    root=Path(root);lock=read(root/'pretest_lock.json')
    if lock['test_predictions_created']!=0:raise ValueError('invalid retrospective lock')
    for name,digest in lock['model_sha256'].items():assert sha256(root/'model'/name/'gate.pt')==digest
    assert sha256(root/'calibration/selection_rules.json')==lock['rules_sha256']
    assert sha256(root/'labels/training_manifest.jsonl')==lock['train_manifest_sha256']
    for path,digest in lock['source_sha256'].items():assert sha256(path)==digest,(path,'source changed after lock')
    for name,digest in lock['confirmation_manifest_sha256'].items():assert sha256(root/'datasets'/f'{name}_confirmation_manifest.jsonl')==digest
    return lock


def assert_confirmation_independence(root=ROOT):
    root=Path(root);used=set(read(root/'audit/previously_used_sources.json'));assigned=set(read(root/'audit/source_roles.json'))
    oldhash=set()
    paths=list((DATA_ROOT/'utility_gate').glob('*/qa_manifest.jsonl'))
    paths += [DATA_ROOT/'relevance_gate/official_s4_expanded_fixed_20261026/causal-manifest.jsonl']
    paths += list((REPO/'artifacts/utility_gate/gate2_grounding_20261027/datasets').glob('*manifest.jsonl'))
    for path in paths:
        for row in read_jsonl(path):
            oldhash.update(row[k] for k in ('media_sha256','source_media_sha256','audio_media_sha256') if row.get(k))
    counts={};sources={}
    for cohort in ('llp','qa'):
        rows=read_jsonl(root/'datasets'/f'{cohort}_confirmation_manifest.jsonl')
        ids={r['source_video_id'].removeprefix('youtube:') for r in rows}
        if ids&(used|assigned):raise AssertionError('confirmation source used before or for development')
        aliases=[r['source_video_id'] for r in rows if r['media_sha256'] in oldhash]
        if aliases:raise AssertionError('confirmation byte alias of prior media')
        counts[cohort]={'sources':len(ids),'rows':len(rows),'prior_source_overlap':0,'training_calibration_overlap':0,'prior_byte_aliases':0}
        sources[cohort]=ids
    if sources['llp']&sources['qa']:raise AssertionError('two confirmation datasets share sources')
    write_json(root/'audit/confirmation_independence.json',counts)


def correspondence_summary(records,rules,root=ROOT):
    root=Path(root);summary={}
    training=[r for r in read_jsonl(root/'labels/training_manifest.jsonl') if r['split']=='train' and r['kind']=='LLP' and r['quality_valid'] and r['corr_target'] is not None]
    train_prevalence=float(np.mean([r['corr_target'] for r in training]))
    for grid in ('1s','2s'):
        allrows=[r for r in records if r['grid']==grid]
        available=[r for r in allrows if r['available']]
        primary=[r for r in available if r['proxy_target'] is not None]
        methods={**{name:('u' if name=='utility_only' else 'r') for name in MODES},
            'constant_prevalence':'constant','always_off':'off','always_on':'on'}
        result={'scheduled_rows':len(allrows),'available_rows':len(available),'primary_known_rows':len(primary),
            'sources':len({r['source_video_id'] for r in allrows}),
            'unknown_counts':dict(Counter(r['label_strength'] for r in allrows if r['proxy_target'] is None)),
            'quality_counts':dict(Counter(r['quality'] for r in allrows)),'methods':{}}
        for name,head in methods.items():
            method={}
            for target,rows in [('primary_same_bin_vs_proxy_negative',primary),('all_current_bin_joint_diagnostic',available)]:
                y=[r['proxy_target'] if target.startswith('primary') else r['current_bin_joint'] for r in rows]
                if head in ('constant','off','on'):
                    score=[train_prevalence if head=='constant' else 0.]*len(rows)
                    decision=[train_prevalence>.5 if head=='constant' else head=='on']*len(rows)
                else:
                    score=[r['scores'][name][head] for r in rows]
                    threshold=float(rules[name]['utility_threshold'] if head=='u' else rules[name]['corr_threshold'])
                    decision=[s>threshold for s in score]
                ids=[r['source_video_id'] for r in rows]
                value=binary_cluster_ci(y,score,decision,ids,repetitions=2000,seed=SEED)
                value['prevalence']=float(np.mean(y)) if len(y) else None
                value['score_interpretation']='raw joint-event proxy logit' if head=='r' else 'utility diagnostic, not correspondence' if head=='u' else 'constant control'
                grouped=defaultdict(list)
                for yi,si,sid in zip(y,score,ids):grouped[sid].append((yi,si))
                differences=[];paired=[]
                for sid,g in grouped.items():
                    pos=[s for v,s in g if v];neg=[s for v,s in g if not v]
                    if pos and neg:differences.append(float(np.mean(pos)-np.mean(neg)));paired.append(sid)
                value['within_source_positive_minus_negative_score']=paired_source_interval(differences,paired)
                method[target]=value
            method['by_subtype']={category:{'rows':len(group),'sources':len({r['source_video_id'] for r in group}),
                'mean_score':float(np.mean([r['scores'][name][head] for r in group])) if head in ('r','u') else 0.}
                for category in ('joint_same_class','different_categories','audio_only','visual_only','neither')
                if (group:=[r for r in available if r['subtype']==category])}
            result['methods'][name]=method
        # Paired source resampling uses the same videos for both methods.
        y=np.asarray([r['proxy_target'] for r in primary],bool)
        groups=sorted({r['source_video_id'] for r in primary})
        indices=[np.flatnonzero(np.asarray([r['source_video_id'] for r in primary])==g) for g in groups]
        comparisons={}
        for dual,single in [('dual_llp','corr_only_llp'),('dual_combined','corr_only_combined')]:
            d=np.asarray([r['scores'][dual]['r'] for r in primary]);s=np.asarray([r['scores'][single]['r'] for r in primary])
            if len(groups)<2 or len(np.unique(y))<2:
                comparisons[dual+' minus '+single]={'status':'insufficient_data','n':len(y),'sources':len(groups)}
                continue
            samples={'AUROC':[],'AUPRC':[]};rng=np.random.default_rng(SEED)
            for _ in range(2000):
                index=np.concatenate([indices[j] for j in rng.integers(0,len(groups),len(groups))])
                if y[index].any() and (~y[index]).any():
                    samples['AUROC'].append(float(roc_auc_score(y[index],d[index])-roc_auc_score(y[index],s[index])))
                    samples['AUPRC'].append(float(average_precision_score(y[index],d[index])-average_precision_score(y[index],s[index])))
            comparisons[dual+' minus '+single]={k:{'difference':float((roc_auc_score if k=='AUROC' else average_precision_score)(y,d)-(roc_auc_score if k=='AUROC' else average_precision_score)(y,s)),
                'ci95':np.quantile(v,[.025,.975]).tolist(),'repetitions':2000,'sources':len(groups),'n':len(y)} for k,v in samples.items()}
        result['paired_differences']=comparisons
        summary[grid]=result
    write_json(root/'evaluation/correspondence_summary.json',summary)
    return summary


@torch.inference_mode()
def evaluate_llp(runtime,models,rules,root=ROOT):
    root=Path(root);verify_lock(root)
    features=llp_features(runtime,'confirmation',root)
    official=Path(read(root/'sources/official_repositories.json')['YapengTian/AVVP-ECCV20']['path'])/'data'
    a=read_intervals(official/'AVVP_eval_audio.csv',False);v=read_intervals(official/'AVVP_eval_visual.csv',False)
    output=root/'evaluation/LLP_new_confirmation_predictions.jsonl'
    completed=root/'evaluation/LLP_completed_sources.jsonl'
    prior=read_jsonl(output) if output.exists() else [];done_rows={(r['clip_id'],r['grid'],r['time_s']) for r in prior}
    done={r['clip_id'] for r in read_jsonl(completed)} if completed.exists() else set()
    for i,row in enumerate(features,1):
        if row['clip_id'] in done:continue
        cached=torch.load(row['features_path'],map_location='cpu',weights_only=True)
        for grid_name,grid in cached['grids'].items():
            x=load_feature_payload(grid['features']);scores={name:predict(model,x) for name,model in models.items()}
            for j,t in enumerate(grid['times']):
                if (row['clip_id'],grid_name,t) in done_rows:continue
                label=labels_at(a.get(row['clip_id'],[]),v.get(row['clip_id'],[]),t);target,strength,_=dense_target(label)
                append_jsonl(output,{'clip_id':row['clip_id'],'source_video_id':row['source_video_id'],'official_split':'test',
                    'grid':grid_name,'time_s':t,'frame_pts_s':grid['frame_pts_s'][j],'audio_start_s':max(0,t-2),'audio_end_s':t,
                    'available':bool(x.valid_audio[j]),'quality':grid['quality'][j],**label,'proxy_target':target,'label_strength':strength,
                    'scores':{name:{'r':float(values[0][j]),'u':float(values[1][j])} for name,values in scores.items()},
                    'evaluation_labels_passed_to_gate':False})
        append_jsonl(completed,{'clip_id':row['clip_id']})
        if i%16==0:print({'LLP_new_confirmation_scored':i},flush=True)
    return correspondence_summary(read_jsonl(output),rules,root)


def downstream_summary(records,windows,root=ROOT):
    root=Path(root);methods=sorted(records[0]['branches'])
    ids=[r['source_video_id'] for r in records];vision=np.array([r['branches']['always_off']['correct'] for r in records],int)
    vision_nll=np.array([r['branches']['always_off']['gold_nll'] for r in records])
    summary={'questions':len(records),'sources':len(set(ids)),
        'by_public_category':dict(Counter(r['question_modality'] for r in records)),
        'methods':{},'interpretation':'new source-independent QA; separately trained stage1 vs stage3 remains a policy/checkpoint confound'}
    for name in methods:
        correct=np.array([r['branches'][name]['correct'] for r in records],int)
        nll=np.array([r['branches'][name]['gold_nll'] for r in records])
        summary['methods'][name]={'correct':int(correct.sum()),'n':len(correct),'accuracy':mean_ci(correct,ids,seed=SEED),
            'gold_NLL':mean_ci(nll,ids,seed=SEED),'paired_accuracy_minus_vision':mean_ci(correct-vision,ids,seed=SEED),
            'paired_NLL_minus_vision':mean_ci(nll-vision_nll,ids,seed=SEED),
            'visual_only_wins':int(((vision==1)&(correct==0)).sum()),'method_only_wins':int(((vision==0)&(correct==1)).sum()),
            'mean_audio_activation':float(np.mean([r['activation'][name] for r in records])),
            'cached_feature_LLM_timing_mean_s':{k:float(np.mean([r['branches'][name][k] for r in records]))
                for k in ('llm_prefill_s','llm_ttft_s','llm_total_s')}}
    valid=[r for r in windows if r['available']];target=np.array([r['target_utility'] for r in valid]);wids=[r['source_video_id'] for r in valid]
    utility={}
    for name in ('utility_only','dual_llp','dual_combined','constant_zero','constant_train_median'):
        predicted=np.array([r['u_predictions'][name] for r in valid]);error=abs(predicted-target)
        utility[name]={'windows':len(valid),'sources':len(set(wids)),'MAE':mean_ci(error,wids,seed=SEED),
            'Spearman':spearman_source_interval(predicted,target,wids,seed=SEED),
            'paired_MAE_minus_zero':mean_ci(error-abs(target),wids,seed=SEED),
            'paired_MAE_minus_utility_only':mean_ci(error-abs(np.array([r['u_predictions']['utility_only'] for r in valid])-target),wids,seed=SEED)}
    summary['independent_window_utility_regression']=utility
    summary['paired_policy_comparisons']={}
    for dual in ('dual_llp','dual_combined'):
        for baseline in ('utility_only','corr_only_combined','constant_utility','always_off','original_AudioRouter'):
            accuracy=[int(r['branches'][dual]['correct'])-int(r['branches'][baseline]['correct']) for r in records]
            nll=[r['branches'][dual]['gold_nll']-r['branches'][baseline]['gold_nll'] for r in records]
            summary['paired_policy_comparisons'][dual+' minus '+baseline]={'accuracy':mean_ci(accuracy,ids,seed=SEED),'NLL':mean_ci(nll,ids,seed=SEED)}
    summary['same_checkpoint_real_minus_null']={
        'accuracy':mean_ci([int(r['branches']['original_AudioRouter']['correct'])-int(r['branches']['same_stage3_null_audio']['correct']) for r in records],ids,seed=SEED),
        'NLL':mean_ci([r['branches']['original_AudioRouter']['gold_nll']-r['branches']['same_stage3_null_audio']['gold_nll'] for r in records],ids,seed=SEED)}
    write_json(root/'downstream/summary.json',summary)
    return summary


@torch.inference_mode()
def evaluate_qa(runtime,models,rules,root=ROOT):
    root=Path(root);verify_lock(root)
    path=root/'downstream/new_QA_predictions.jsonl';windowpath=root/'evaluation/new_QA_window_targets.jsonl'
    prior=read_jsonl(path) if path.exists() else [];done={r['question_id'] for r in prior}
    prior_windows={(r['question_id'],r['window']) for r in read_jsonl(windowpath)} if windowpath.exists() else set()
    old=REPO/'artifacts/utility_gate/gate2_20261027'
    frozen_meta=read(old/'temporal_calibration/streaming_threshold.json');frozen,_=load_gate2(frozen_meta['model_checkpoint'],runtime.device)
    train=[r for r in read_jsonl(root/'labels/training_manifest.jsonl') if r['kind']=='Utility' and r['split']=='train' and r['quality_valid']]
    median=float(np.median([r['utility_target'] for r in train]));mean=float(np.mean([r['utility_target'] for r in train]))
    rate=rules['dual_combined']['dual_validation']['activation']
    for i,row in enumerate(read_jsonl(root/'datasets/qa_confirmation_manifest.jsonl')):
        if row['question_id'] in done:continue
        if row['official_split']!='evaluation' or row['split']!='test':raise ValueError('confirmation QA role')
        cache=DATA/'features/qa_confirmation'/(row['videoID']+'.pt')
        if cache.exists():payload=torch.load(cache,map_location='cpu',weights_only=True)
        else:
            payload=runtime.encode(row);payload.update(split='test',source_video_id=row['source_video_id']);save_tensors(cache,payload)
        x=features_from_cache(payload);scores={name:predict(model,x) for name,model in models.items()}
        valid=payload['audio_valid'].numpy();audio,_=qualified_latents(runtime,payload);audio=audio[0]
        vision=payload['vision_latents'].to(runtime.device)
        null,_=runtime.audio_router(payload['visual'].to(runtime.device),torch.zeros_like(payload['causal_audio']).to(runtime.device),payload['timestamps'].to(runtime.device))
        choices={name:apply_rule(r,u,valid,rules[name]) for name,(r,u) in scores.items()}
        frozen_score=frozen.predict(x.to(runtime.device)).utility_score.detach().cpu().numpy()
        choices.update(always_off=np.zeros(len(x.visual),bool),qualified_audio=valid.copy(),
            constant_utility=valid&(mean>0),random_rate_matched=(np.random.default_rng(SEED+i).random(len(valid))<rate)&valid,
            frozen_Gate2=(frozen_score>frozen_meta['threshold'])&valid)
        branches={};activation={};prompt=runtime.prompt(row)
        for name,choice in choices.items():
            z=torch.where(torch.tensor(choice,device=runtime.device)[:,None,None],audio,vision)
            assert z.shape[1]==64
            branches[name]=score_answer(runtime.generate(prompt,z),row['answer'],row['dataset']);activation[name]=float(choice.mean())
        branches['original_AudioRouter']=score_answer(runtime.generate(prompt,payload['audio_latents'].to(runtime.device)),row['answer'],row['dataset'])
        activation['original_AudioRouter']=1.
        branches['same_stage3_null_audio']=score_answer(runtime.generate(prompt,null),row['answer'],row['dataset']);activation['same_stage3_null_audio']=1.
        equivalent={}
        for name,choice in choices.items():
            signature=tuple(choice.tolist())
            if signature in equivalent:assert branches[name]['generated_ids']==branches[equivalent[signature]]['generated_ids']
            else:equivalent[signature]=name
        baseline=branches['qualified_audio'];windows=[]
        for j,t in enumerate(payload['timestamps'].tolist()):
            z=audio.clone();z[j]=vision[j];identical=torch.equal(z,audio)
            off=baseline if identical else score_answer(runtime.generate(prompt,z),row['answer'],row['dataset'])
            windows.append({'question_id':row['question_id'],'source_video_id':row['source_video_id'].removeprefix('youtube:'),
                'split':'test','official_split':'evaluation','window':j,'time_s':t,'available':bool(valid[j]),
                'target_utility':off['gold_nll']-baseline['gold_nll'],'accuracy_utility':int(baseline['correct'])-int(off['correct']),
                'u_predictions':{**{name:float(scores[name][1][j]) for name in ('utility_only','dual_llp','dual_combined')},
                    'constant_zero':0.,'constant_train_median':median},
                'actual_counterfactual_generation':not identical,'never_used_to_fit_or_calibrate':True})
        for window in windows:
            if (window['question_id'],window['window']) not in prior_windows:append_jsonl(windowpath,window)
        append_jsonl(path,{'question_id':row['question_id'],'source_video_id':row['source_video_id'].removeprefix('youtube:'),
            'question_modality':row['question_modality'],'media_sha256':row['media_sha256'],'cache_path':str(cache),
            'branches':branches,'activation':activation,'valid_audio_windows':int(valid.sum()),'total_windows':len(valid),
            'raw_head_scores':{name:{'r':r.tolist(),'u':u.tolist(),'selected':choices[name].tolist()} for name,(r,u) in scores.items()},
            'all_policy_branches_actually_generated':True,'identical_policy_generated_ID_parity':True,
            'feature_extraction_timing':payload['timing'],
            'visual_token_budget_per_frame':64,'question_passed_to_gate':False})
        if (i+1)%8==0:print({'new_QA_confirmation_completed':i+1},flush=True)
    return downstream_summary(read_jsonl(path),read_jsonl(windowpath),root)


def run(root=ROOT):
    root=Path(root);seed_all(SEED);verify_lock(root);assert_confirmation_independence(root)
    runtime=StrictStreamingRuntime()
    models={name:load_dual_head(root/'model'/name/'gate.pt',runtime.device)[0] for name in MODES}
    rules=read(root/'calibration/selection_rules.json')
    write_json(root/'evaluation/runtime.json',runtime.metadata)
    evaluate_llp(runtime,models,rules,root);evaluate_qa(runtime,models,rules,root)
    verify_lock(root)
    print({'new_independent_confirmation_evaluation':'completed'},flush=True)


if __name__=='__main__':run()
