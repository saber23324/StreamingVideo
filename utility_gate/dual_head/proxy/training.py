"""Joint masked optimization on distinct authorized label sources."""
from __future__ import annotations

from collections import Counter
import copy
from dataclasses import replace
from datetime import datetime, timezone
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
import torch

from utility_gate.common import read_jsonl, seed_all, sha256, write_json, write_jsonl
from utility_gate.features import GateFeatures, load_feature_payload, take_features
from utility_gate.dual_head.losses import DualSupervision, dual_loss
from utility_gate.dual_head.model import DualHeadGate
from .protocol import OLD, ROOT, SEED, read

MODES={
    'corr_only_llp':(['LLP'],False),
    'corr_only_robust':(['AVS_original','AVS_synthetic'],False),
    'corr_only_combined':(['LLP','AVS_original','AVS_synthetic'],False),
    'utility_only':([],True),
    'dual_llp':(['LLP'],True),
    'dual_combined':(['LLP','AVS_original','AVS_synthetic'],True)}


def guard_rows(rows,split,root=ROOT):
    root=Path(root)
    if split not in ('train','val') or not rows:raise ValueError('nonempty development role required')
    protocol=read(root/'protocol.json')
    if 'LLP official Validation' not in protocol['authorization']:raise ValueError('explicit validation-repurposing protocol required')
    roles=read(root/'audit/source_roles.json')
    for row in rows:
        if row['split']!=split:raise ValueError('training/calibration role contamination')
        if row['kind'] not in ('LLP','AVS_original','AVS_synthetic','Utility'):raise ValueError('unregistered label source')
        if row['official_split']!=('val' if row['kind']=='LLP' else 'train'):raise ValueError('official Test/evaluation cannot train or calibrate')
        for sid in (row['source_video_id'],row.get('audio_source_video_id',row['source_video_id'])):
            if roles.get(sid)!=split:raise ValueError('source/donor assigned to another role')
        if row.get('corr_target') is not None:
            if row['kind']=='Utility':raise ValueError('QA utility cannot become correspondence label')
            if row['corr_target'] not in (0.,1.):raise ValueError('binary proxy target required')
            if row['corr_target']==0 and 'proxy_negative' not in row['label_strength']:raise ValueError('negative needs explicit proxy provenance')
        if row.get('utility_target') is not None:
            if row['kind']!='Utility' or 'measured_' not in row['label_strength']:raise ValueError('only measured window QA utility may supervise utility head')
            if not math.isfinite(row['utility_target']):raise ValueError('finite utility label required')


def assert_isolated(rows):
    identities={}
    for row in rows:
        keys=[('source',row['source_video_id']),('source',row.get('audio_source_video_id',row['source_video_id']))]
        keys += [('bytes',row[k]) for k in ('media_sha256','visual_media_sha256','audio_media_sha256') if row.get(k)]
        for key in keys:
            if key in identities and identities[key]!=row['split']:raise ValueError('cross-role source/donor/media alias')
            identities[key]=row['split']


def load_features(rows):
    cache={};chunks=[]
    for row in rows:
        path=row['features_path']
        if path not in cache:cache[path]=torch.load(path,map_location='cpu',weights_only=True)
        value=cache[path]
        if value['split']!=row['split'] or value['source_video_id'].removeprefix('youtube:')!=row['source_video_id']:
            raise ValueError('feature role/source mismatch')
        feature=load_feature_payload(value['grids'][row['grid']]['features'])
        selected=take_features(feature,slice(row['position'],row['position']+1))
        if len(selected.visual)!=1 or bool(selected.valid_audio[0])!=row['quality_valid']:raise ValueError('feature/label index or availability mismatch')
        chunks.append(selected)
    return GateFeatures(*(torch.cat([getattr(v,k) for v in chunks]) for k in ('visual','audio','context','valid_audio')))


def make_labels(rows):
    corpus=lambda r:'AVS' if r['kind'].startswith('AVS') else r['kind']
    counts=Counter((corpus(r),r['source_video_id']) for r in rows if r['quality_valid'])
    source_counts={k:len({r['source_video_id'] for r in rows if corpus(r)==k and r['quality_valid']}) for k in ('LLP','AVS','Utility')}
    c=[];cm=[];cw=[];u=[];um=[];uw=[]
    for r in rows:
        available=r['quality_valid'];hasc=r['corr_target'] is not None;hasu=r.get('utility_target') is not None
        denom=max(counts[corpus(r),r['source_video_id']],1)*max(source_counts[corpus(r)],1)
        c.append(r['corr_target'] if hasc else float('nan'));cm.append(available and hasc);cw.append(r.get('corr_weight',0.)/denom)
        u.append(r['utility_target'] if hasu else float('nan'));um.append(available and hasu);uw.append(1/denom)
    f=lambda v:torch.tensor(v,dtype=torch.float32)[:,None]
    return DualSupervision(f(c),torch.tensor(cm,dtype=torch.bool)[:,None],f(cw),f(u),torch.tensor(um,dtype=torch.bool)[:,None],f(uw))


def take_labels(value,index):return DualSupervision(*(getattr(value,k)[index] for k in value.__dataclass_fields__))


def training_by_source(rows,output):
    result={}
    c=output.correspondence_logit[:,0].detach().numpy();u=output.utility_delta[:,0].detach().numpy()
    for kind in sorted({r['kind'] for r in rows}):
        indexes=[i for i,r in enumerate(rows) if r['kind']==kind and r['quality_valid']]
        corr=[i for i in indexes if rows[i]['corr_target'] is not None]
        util=[i for i in indexes if rows[i].get('utility_target') is not None]
        values={'all_rows':sum(r['kind']==kind for r in rows),'eligible_rows':len(indexes),
            'sources':len({rows[i]['source_video_id'] for i in indexes}),
            'unknown_corr_rows':sum(rows[i]['corr_target'] is None for i in indexes)}
        if corr:
            y=np.asarray([rows[i]['corr_target'] for i in corr]);s=c[corr]
            values.update(corr_rows=len(corr),positive=int(y.sum()),proxy_negative=int((1-y).sum()),
                BCE=float(np.mean(np.logaddexp(0,s)-y*s)),
                AUROC=float(roc_auc_score(y,s)) if len(set(y))==2 else None,
                AUPRC=float(average_precision_score(y,s)) if len(set(y))==2 else None)
        if util:values.update(utility_rows=len(util),MAE=float(np.mean(abs(u[util]-[rows[i]['utility_target'] for i in util]))))
        result[kind]=values
    return result


def encode_threshold(value):
    return '-inf' if value==-np.inf else 'inf' if value==np.inf else float(value)


def threshold_candidates(values):
    values=np.asarray(values,float)
    unique=np.unique(values)
    mids=(unique[:-1]+unique[1:])/2
    if len(mids)>11:mids=mids[np.unique(np.linspace(0,len(mids)-1,11).astype(int))]
    return np.unique(np.r_[-np.inf,mids,0.,np.inf]) if len(values) else np.array([-np.inf,np.inf])


def calibrate(model,rows,features,corr_kinds,utility_active):
    with torch.inference_mode():
        outputs=[model.forward_features(take_features(features,slice(i,i+1))) for i in range(len(features.visual))]
    c=np.asarray([float(o.correspondence_logit[0,0]) for o in outputs]);u=np.asarray([float(o.utility_delta[0,0]) for o in outputs])
    ci=[i for i,r in enumerate(rows) if r['kind'] in corr_kinds and r['quality_valid'] and r['corr_target'] is not None]
    ui=[i for i,r in enumerate(rows) if r['kind']=='Utility' and r['quality_valid']]
    ct=np.inf;ut=np.inf;cbest=None;ubest=None
    if ci:
        y=np.asarray([rows[i]['corr_target'] for i in ci])
        trials=[{'threshold':encode_threshold(t),'balanced_accuracy':float(balanced_accuracy_score(y,c[ci]>t)),
                 'activation':float(np.mean(c[ci]>t))} for t in threshold_candidates(c[ci])]
        cbest=min(trials,key=lambda r:(-r['balanced_accuracy'],r['activation']));ct=float(cbest['threshold'])
    if utility_active and ui:
        target=np.asarray([rows[i]['utility_target'] for i in ui]);groups=Counter(rows[i]['source_video_id'] for i in ui)
        weight=np.asarray([1/groups[rows[i]['source_video_id']] for i in ui]);weight/=weight.sum()
        trials=[]
        for t in threshold_candidates(u[ui]):
            accepted=u[ui]>t
            trials.append({'threshold':encode_threshold(t),'conditional_NLL_gain':float(np.sum(weight*target*accepted)),
                           'activation':float(accepted.mean())})
        ubest=min(trials,key=lambda r:(-r['conditional_NLL_gain'],r['activation']));ut=float(ubest['threshold'])
    rule={'mode':'utility' if not corr_kinds else 'corr' if not utility_active else 'dual',
          'corr_threshold':encode_threshold(ct),'utility_threshold':encode_threshold(ut),
          'corr_validation':cbest,'utility_validation':ubest,'fit_role':'val',
          'score_scope':'joint-event/synthetic pairing proxy; not a true visibility probability'}
    if corr_kinds and utility_active and ui:
        target=np.asarray([rows[i]['utility_target'] for i in ui]);groups=Counter(rows[i]['source_video_id'] for i in ui)
        weight=np.asarray([1/groups[rows[i]['source_video_id']] for i in ui]);weight/=weight.sum()
        candidates=[]
        for rt in threshold_candidates(c[ui]):
            for qt in threshold_candidates(u[ui]):
                accepted=(c[ui]>rt)&(u[ui]>qt)
                candidates.append({'corr_threshold':encode_threshold(rt),'utility_threshold':encode_threshold(qt),
                    'conditional_NLL_gain':float(np.sum(weight*target*accepted)),'activation':float(accepted.mean())})
        best=min(candidates,key=lambda r:(-r['conditional_NLL_gain'],r['activation']))
        rule['selection_corr_threshold']=best['corr_threshold'];rule['selection_utility_threshold']=best['utility_threshold']
        rule['dual_validation']=best
    return rule


def fit_all(root=ROOT):
    root=Path(root)
    if (root/'pretest_lock.json').exists():raise FileExistsError('model/threshold fitting is closed after the preprediction lock')
    if (root/'evaluation/LLP_new_confirmation_predictions.jsonl').exists() or (root/'downstream/new_QA_predictions.jsonl').exists():
        raise ValueError('cannot fit after seeing confirmation predictions')
    rows=read_jsonl(root/'labels/training_manifest.jsonl')
    for split in ('train','val'):guard_rows([r for r in rows if r['split']==split],split,root)
    assert_isolated(rows)
    scale=max(float(np.quantile([abs(r['utility_target']) for r in rows if r['split']=='train' and r['kind']=='Utility' and r['quality_valid']],.75)),1e-4)
    config={**read(root/'protocol.json')['fit'],'target_scale':scale,'training_manifest_sha256':sha256(root/'labels/training_manifest.jsonl'),
        'new_model_initialization':'fresh same seed per mode','large_models_loaded_for_fitting':False,
        'feature_sha256':{r['features_path']:sha256(r['features_path']) for r in rows},
        'label_tasks_need_not_coexist_on_same_row':True,'joint_labeled_rows':sum(r['corr_target'] is not None and r.get('utility_target') is not None for r in rows)}
    write_json(root/'model/training_configuration.json',config)
    rules={}
    for mode,(corr_kinds,hasutility) in MODES.items():
        folder=root/'model'/mode
        if (folder/'gate.pt').exists():
            rules[mode]=read(folder/'rule.json');continue
        selected=[r for r in rows if r['kind'] in corr_kinds or (r['kind']=='Utility' and hasutility)]
        train=[r for r in selected if r['split']=='train'];val=[r for r in selected if r['split']=='val']
        tx,vx=load_features(train),load_features(val);ty,vy=make_labels(train),make_labels(val)
        seed_all(SEED);model=DualHeadGate(utility_scale=scale)
        model.correspondence_head.requires_grad_(bool(corr_kinds));model.utility_head.requires_grad_(hasutility);model.decision_head.requires_grad_(False)
        initial={k:v.clone() for k,v in model.state_dict().items()}
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['lr'],weight_decay=config['weight_decay'])
        generator=torch.Generator().manual_seed(SEED);best=None;history=[]
        for epoch in range(config['epochs']):
            model.train();losses=[]
            for index in torch.randperm(len(train),generator=generator).split(config['batch_size']):
                optimizer.zero_grad(set_to_none=True)
                loss,info=dual_loss(model.forward_features(take_features(tx,index)),take_labels(ty,index),float(bool(corr_kinds)),float(hasutility),scale)
                if not info['corr_active'] and not info['utility_active']:continue
                if not torch.isfinite(loss):raise ValueError('nonfinite fit loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
            model.eval()
            with torch.inference_mode():vl,vi=dual_loss(model.forward_features(vx),vy,float(bool(corr_kinds)),float(hasutility),scale)
            history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'validation_loss':float(vl),**vi})
            if best is None or float(vl)<best['loss']:best={'epoch':epoch+1,'loss':float(vl),'state':copy.deepcopy(model.state_dict())}
        model.load_state_dict(best['state']);model.eval()
        model.head_status={'correspondence':'trained_joint_event_and_synthetic_proxy' if corr_kinds else 'untrained',
                           'utility':'trained_regression' if hasutility else 'untrained','decision':'untrained'}
        changed=[k for k,v in model.state_dict().items() if not torch.equal(v,initial[k])]
        if corr_kinds and not any(k.startswith('correspondence_head') for k in changed):raise AssertionError('correspondence head did not learn')
        if hasutility and not any(k.startswith('utility_head') for k in changed):raise AssertionError('utility head did not learn')
        folder.mkdir(exist_ok=True)
        torch.save({'kind':'audiorouter_dual_head','spec':model.spec,'state_dict':model.state_dict(),'head_status':model.head_status,
            'provenance':{'protocol':str(root/'protocol.json'),'mode':mode,'training_manifest_sha256':config['training_manifest_sha256'],
                          'LLP_train_official_split':'Validation','no_official_Test_training_or_calibration':True},'best_epoch':best['epoch']},folder/'gate.pt')
        rule=calibrate(model,val,vx,corr_kinds,hasutility);rules[mode]=rule
        write_json(folder/'rule.json',rule);write_json(folder/'history.json',history)
        with torch.inference_mode():tr=model.forward_features(tx);va=model.forward_features(vx)
        write_json(folder/'development_metrics.json',{'train':training_by_source(train,tr),'val':training_by_source(val,va),
            'train_by_label_strength':training_by_source([{**r,'kind':r['kind']+':'+r['label_strength']} for r in train],tr),
            'val_by_label_strength':training_by_source([{**r,'kind':r['kind']+':'+r['label_strength']} for r in val],va),
            'parameters_total':sum(p.numel() for p in model.parameters()),'parameters_trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'best_epoch':best['epoch'],'changed_parameters':changed,'heads_receive_disjoint_supervision':True})
        write_jsonl(folder/'validation_scores.jsonl',[{'row_id':r['row_id'],'kind':r['kind'],'source_video_id':r['source_video_id'],
            'available':r['quality_valid'],'corr_target':r['corr_target'],'utility_target':r.get('utility_target'),
            'r_logit':float(va.correspondence_logit[i,0]),'u_delta':float(va.utility_delta[i,0])} for i,r in enumerate(val)])
        print({'trained':mode,'best_epoch':best['epoch'],'train_rows':len(train),'validation_rows':len(val)},flush=True)
    write_json(root/'calibration/selection_rules.json',rules)
    # This lock is written before confirmation features/scores or QA outputs.
    source_paths=list((Path(__file__).parent).glob('*.py'))
    write_json(root/'pretest_lock.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'model_sha256':{name:sha256(root/'model'/name/'gate.pt') for name in MODES},
        'rules_sha256':sha256(root/'calibration/selection_rules.json'),'protocol_sha256':sha256(root/'protocol.json'),
        'train_manifest_sha256':config['training_manifest_sha256'],
        'confirmation_manifest_sha256':{name:sha256(root/'datasets'/f'{name}_confirmation_manifest.jsonl') for name in ('llp','qa')},
        'source_sha256':{str(p):sha256(p) for p in source_paths},'test_predictions_created':0,
        'old_inspected_Test_used_to_fit':False,'validation_only_model_epoch_and_threshold_selection':True})
    print({'all_six_models_trained_and_locked':True},flush=True)


if __name__=='__main__':fit_all()
