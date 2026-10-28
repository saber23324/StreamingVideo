"""Guarded small-head fitting; semantic modes refuse one-class correspondence."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from utility_gate.common import read_jsonl, seed_all, sha256, write_json, write_jsonl
from utility_gate.features import take_features
from utility_gate.gate2_interfaces import load_gate2
from utility_gate.grounding.stats import mean_ci
from .calibration import validate_calibration_rows
from .data_audit import GATE2, ROOT, read
from .datasets import assert_source_isolation, labels_for_rows, load_features, training_support, validate_rows
from .diagnostics import paired_source_interval, spearman_source_interval
from .losses import DualSupervision, dual_loss
from .model import DualHeadGate


def take_labels(labels,index):
    return DualSupervision(*(getattr(labels,k)[index] for k in labels.__dataclass_fields__))


def regression_report(predictions,rows):
    chosen=np.asarray([r['eligible_audio'] for r in rows],bool)
    target=np.asarray([r['nll_utility'] for r in rows])[chosen]
    prediction=np.asarray(predictions)[chosen]
    sources=[r['source_video_id'] for r in rows if r['eligible_audio']]
    return {'eligible_rows':len(target),'sources':len(set(sources)),'MAE':mean_ci(abs(prediction-target),sources),
            'Spearman':spearman_source_interval(prediction,target,sources),
            'interpretation':'conditional-window utility regression on development data, not actual mixed-policy QA accuracy'}


def fit(mode='dual_head',root=ROOT,epochs=20):
    root=Path(root)
    u=[r for r in read_jsonl(root/'training_assets/utility_train_val.jsonl') if r['supervision_scope']=='measured_window']
    c=read_jsonl(root/'training_assets/correspondence_train_val.jsonl')
    candidates=u if mode=='utility_only' else c if mode=='corr_only' else c+u
    train=[r for r in candidates if r['dataset_role']=='train'];val=[r for r in candidates if r['dataset_role']=='val']
    forbidden=set().union(*(set(v) for v in read(root/'audit/evaluation_source_registry.json').values()))
    validate_rows(train,'train',forbidden);validate_rows(val,'val',forbidden);assert_source_isolation(train,val)
    support={'train':training_support(train,mode),'val':training_support(val,mode)}
    output=root/'model'/mode
    if output.exists(): raise FileExistsError('head experiment output is immutable; choose a new run')
    output.mkdir()
    if any(x['status']!='eligible_for_fitting' for x in support.values()):
        write_json(output/'training_decision.json',{'status':'not_run','support':support,'test_data_used':False,
            'reason':'semantic training fails documented supervision preconditions; unknowns remain masked'})
        print({'mode':mode,'status':'not_run','support':support});return
    seed_all(1234)
    # Reuse the architecture/features, but fit fresh weights after the strict
    # source exclusions. Old Gate2 weights remain a frozen comparison only.
    reference=read(GATE2/'temporal_model/selection.json')['primary']['checkpoint']
    train_values=[abs(r['nll_utility']) for r in train if r.get('target_family')=='utility' and r['eligible_audio']]
    scale=max(float(np.quantile(train_values,.75)),1e-4) if train_values else 1.
    model=DualHeadGate(utility_scale=scale)
    model.provenance={'initialization':'fresh_seed_1234','old_Gate2_used_for_frozen_comparison_only':True}
    corr_lambda=0. if mode=='utility_only' else 1.
    utility_lambda=0. if mode=='corr_only' else 1.
    model.correspondence_head.requires_grad_(corr_lambda>0)
    model.utility_head.requires_grad_(utility_lambda>0)
    model.decision_head.requires_grad_(False)
    parameters=sum(p.numel() for p in model.parameters())
    if parameters>2_000_000: raise ValueError('declared small-gate budget exceeded')
    train_x,val_x=load_features(train),load_features(val)
    train_y,val_y=labels_for_rows(train),labels_for_rows(val)
    config={'mode':mode,'seed':1234,'epochs':epochs,'optimizer':'AdamW','learning_rate':.001,'weight_decay':.01,
        'batch_size':32,'corr_lambda':corr_lambda,'utility_lambda':utility_lambda,
        'parameters_total':parameters,'parameters_trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'train_source_count':len({r['source_video_id'] for r in train}),'val_source_count':len({r['source_video_id'] for r in val}),
        'initialization':'fresh_seed_1234','frozen_reference_checkpoint':reference,'frozen_reference_sha256':sha256(reference),
        'feature_version':'current/past native Gate2 anchor v1','test_targets_or_features_loaded':False,
        'evaluation_source_IDs_used_for_exclusion_only':True,'utility_scale':model.spec['utility_scale'],
        'utility_scale_fit':'75th percentile absolute eligible training target only; lower bound 1e-4',
        'source_sha256':{str(p):sha256(p) for p in Path(__file__).parent.glob('*.py')},
        'training_asset_sha256':{str(root/'training_assets'/name):sha256(root/'training_assets'/name) for name in ('utility_train_val.jsonl','correspondence_train_val.jsonl')}}
    config['feature_cache_sha256']={r['feature_cache_path']:sha256(r['feature_cache_path']) for r in train+val}
    write_json(output/'train_config.json',config)
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=.01)
    generator=torch.Generator().manual_seed(1234)
    best=None;history=[]
    for epoch in range(epochs):
        model.train();losses=[]
        for index in torch.randperm(len(train),generator=generator).split(32):
            optimizer.zero_grad(set_to_none=True)
            predicted=model.forward_features(take_features(train_x,index))
            loss,_=dual_loss(predicted,take_labels(train_y,index),corr_lambda,utility_lambda,model.spec['utility_scale'])
            if not torch.isfinite(loss): raise ValueError('nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            validation,_=dual_loss(model.forward_features(val_x),val_y,corr_lambda,utility_lambda,model.spec['utility_scale'])
        history.append({'epoch':epoch+1,'training_loss':float(np.mean(losses)),'validation_loss':float(validation)})
        if best is None or float(validation)<best['loss']:
            best={'loss':float(validation),'epoch':epoch+1,'state':copy.deepcopy(model.state_dict())}
    model.load_state_dict(best['state']);model.eval()
    if corr_lambda:model.head_status['correspondence']='trained_semantic'
    else:model.head_status['correspondence']='untrained'
    if utility_lambda:model.head_status['utility']='trained_regression'
    else:model.head_status['utility']='untrained'
    model.head_status['decision']='untrained'
    changed=[k for k,v in model.state_dict().items() if not torch.equal(v,initial[k])]
    if not changed: raise AssertionError('no head/trunk update')
    if mode=='utility_only' and any(k.startswith('correspondence_head') or k.startswith('decision_head') for k in changed):
        raise AssertionError('inactive head changed')
    torch.save({'kind':'audiorouter_dual_head','spec':model.spec,'state_dict':model.state_dict(),
        'head_status':model.head_status,'provenance':{**model.provenance,'mode':mode,'deployment_rule':'NOT_FITTED'},
        'config':config,'best_epoch':best['epoch']},output/'gate.pt')
    write_json(output/'history.json',history)
    write_json(output/'training_decision.json',{'status':'trained_diagnostic_only','head_status':model.head_status,
        'best_epoch':best['epoch'],'changed_parameters':changed,'no_large_model_loaded':True,'not_a_trained_semantic_dual_head':mode!='dual_head'})
    if mode=='utility_only':
        original,_=load_gate2(reference)
        with torch.inference_mode():
            new_scores=model.forward_features(val_x).utility_delta[:,0].numpy()
            frozen_scores=original.predict(val_x).utility_score.numpy()
        median=float(np.median([r['nll_utility'] for r in train if r['eligible_audio']]))
        report={'new_utility_only':regression_report(new_scores,val),'frozen_gate2':regression_report(frozen_scores,val),
            'constant_train_median':regression_report(np.full(len(val),median),val),
            'constant_zero':regression_report(np.zeros(len(val)),val),
            'head_score_correlation':'not_applicable: correspondence head untrained',
            'validation_selection_support':validate_calibration_rows(val),
            'new_test_evaluation':'not_run; no admissible selection calibration'}
        eligible=np.asarray([r['eligible_audio'] for r in val],bool)
        target=np.asarray([r['nll_utility'] for r in val])[eligible]
        sources=[r['source_video_id'] for r in val if r['eligible_audio']]
        report['paired_MAE_new_minus_reference']={name:paired_source_interval(
            abs(new_scores[eligible]-target)-abs(np.asarray(scores)[eligible]-target),sources)
            for name,scores in [('frozen_gate2',frozen_scores),('constant_zero',np.zeros(len(val))),
                                ('constant_train_median',np.full(len(val),median))]}
        write_jsonl(output/'validation_predictions.jsonl',[{
            'row_id':r['row_id'],'source_video_id':r['source_video_id'],'dataset_role':'val',
            'eligible_audio':r['eligible_audio'],'target_nll_utility':r['nll_utility'],
            'predicted_utility_only':float(new_scores[i]),'predicted_frozen_gate2':float(frozen_scores[i]),
            'predicted_train_median':median,'predicted_zero':0.,'new_selection':'not_calibrated'}
            for i,r in enumerate(val)])
        write_json(output/'validation_regression.json',report)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
        for key in ('training_loss','validation_loss'):ax.plot([x['epoch'] for x in history],[x[key] for x in history],label=key)
        ax.set(xlabel='Epoch',ylabel='Masked Huber loss (legal train-only scale)',title='Utility-only diagnostic; correspondence head not trained');ax.legend()
        fig.savefig(output/'training_curve.png',dpi=160);plt.close(fig)
    print(json.dumps({'mode':mode,'best_epoch':best['epoch'],'parameters':parameters,'head_status':model.head_status}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['corr_only','utility_only','dual_head'],default='dual_head')
    p.add_argument('--output',type=Path,default=ROOT);p.add_argument('--epochs',type=int,default=20)
    args=p.parse_args();fit(args.mode,args.output,args.epochs)


if __name__=='__main__':main()
