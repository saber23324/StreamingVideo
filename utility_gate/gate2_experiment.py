"""Gate2 formal-training-source labels, small-gate fits and locked evaluation."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from .common import read_jsonl,sha256,validate_splits,write_json,write_jsonl
from .features import GateFeatures,load_feature_payload,take_features
from .gate2_interfaces import features_from_cache,load_gate2
from .labels import build_labels,complementarity
from .metrics import (random_selector,regression_metrics,selection_metrics,video_bootstrap_gain)
from .train import fit_one


def export_labels(manifest,paired,output):
    original={r['question_id']:r for r in read_jsonl(manifest)}
    rows=read_jsonl(paired)
    if set(original)!={r['question_id'] for r in rows}:
        raise ValueError('incomplete paired QA set')
    labels,config=build_labels(rows)
    output=Path(output)
    statistics={}
    for split in ('train','val','test'):
        selected=[r for r in labels if r['split']==split]
        expected_source='evaluation' if split=='test' else 'train'
        for row in selected:
            source=original[row['question_id']]
            if source['official_split']!=expected_source:
                raise ValueError('official evaluation answers entered a training/validation asset')
            row.update(official_split=source['official_split'],annotation_source_commit=source['annotation_source_commit'],
                supervision_scope='whole legal clip endpoint feasibility proxy; not per-frame utility',
                audio_endpoint=next(x for x in rows if x['question_id']==row['question_id']).get('audio_endpoint','raw legacy audio'))
        directory='evaluation_assets' if split=='test' else 'training_assets'
        write_jsonl(output/directory/(split+'.jsonl'),selected)
        statistics[split]={'n':len(selected),'source':expected_source,
            'label_states':dict(Counter(r['label_state'] for r in selected)),
            'official_modality':dict(Counter(original[r['question_id']]['question_modality'] for r in selected)),
            'parser_failures':sum(not r['parser_valid'] for r in selected),
            'all_audio_invalid':sum(r['audio_missing_fraction']==1 for r in selected),
            'complementarity':complementarity([r for r in rows if r['split']==split])}
    write_json(output/'label_statistics.json',{'by_split':statistics,'label_config':config,
        'paired_sha256':sha256(paired),'qa_manifest_sha256':sha256(manifest),
        'train_and_test_targets_in_different_directories':True,'test_used_to_fit_any_label_threshold':False})
    write_json(output/'label_config.json',config)
    with (output/'complementarity.csv').open('w') as stream:
        values=[{'split':split,**statistics[split]['complementarity']} for split in statistics]
        writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    development=[r for r in rows if r['split'] in ('train','val')]
    decision=complementarity(development)
    (output/'M1_decision.md').write_text('# Gate2 M1 complementarity decision\n\n'
        f'Train/val only: {json.dumps(decision,sort_keys=True)}\n\n'
        'Proceed with the predeclared bounded small-gate comparison; no evaluation answers fit the gate. '
        'The formal-training-source data correct the previous internal-development protocol. '
        'Whole-prefix endpoint utility remains a feasibility proxy. Static vision versus audio/time checkpoint differences remain.\n')
    return statistics


def load_examples(rows,layout):
    if layout not in ('anchor_history','pooled_history'):
        raise ValueError('unknown feature-layout ablation')
    values,energy=[],[]
    for row in rows:
        payload=torch.load(row['feature_cache_path'],map_location='cpu',weights_only=True)
        if payload['cache_key']!=row['feature_cache_key'] or payload['source_video_id']!=row['source_video_id'] or payload['split']!=row['split']:
            raise ValueError('feature source/split/cache identity mismatch')
        value=features_from_cache(payload,layout)
        index=row.get('window_index',len(value.visual)-1)
        current=take_features(value,slice(index,index+1))
        # The two feature layouts use the SAME current-availability policy.
        available=(payload['audio_valid'][index:index+1].bool() if 'window_index' in row else
                   payload['audio_valid'][:index+1].any().reshape(1))
        current=GateFeatures(current.visual,current.audio,current.context,available)
        values.append(current)
        energy.append(float(torch.log1p(payload['audio_rms'][:index+1].mean())))
    result=GateFeatures(*(torch.cat([getattr(v,name) for v in values]) for name in ('visual','audio','context','valid_audio')))
    result.validate()
    return result,np.asarray(energy)


def stable_threshold(rows,scores,valid):
    if not rows or any(r['split']!='val' for r in rows):
        raise ValueError('only validation targets can fit Gate2 thresholds')
    scores=np.asarray(scores,float);valid=np.asarray(valid,bool)
    if scores.shape!=(len(rows),) or valid.shape!=scores.shape or not np.isfinite(scores).all():
        raise ValueError('invalid validation scores')
    unique=np.unique(scores[valid]) if valid.any() else np.asarray([0.])
    margin=max(1.,float(np.abs(unique).max()))
    candidates=np.concatenate(([unique[0]-margin],(unique[:-1]+unique[1:])/2,[unique[-1]+margin]))
    curve=[]
    for threshold in candidates:
        curve.append({'threshold':float(threshold),**selection_metrics(rows,(scores>threshold)&valid)})
    best=min(curve,key=lambda r:(-r['accuracy'],r['gold_nll'],r['gate_on_rate'],-r['threshold']))
    return {'threshold':best['threshold'],'fit_split':'val','comparison':'strict_greater_than',
        'validation':best,'curve':curve,'boundary_policy':'midpoints between distinct validation scores; no test fitting',
        'selection_objective':'validation QA accuracy, then gold NLL, then lower activation'}


def train_gate2(train_file,val_file,output,epochs=60,segment=False):
    train_rows,val_rows=read_jsonl(train_file),read_jsonl(val_file)
    for expected,rows in [('train',train_rows),('val',val_rows)]:
        if not rows or any(r['split']!=expected or r.get('official_split')!='train' for r in rows):
            raise ValueError('training assets must contain only the assigned official-training-source targets')
    unique={(r['question_id'],r['split']):r for r in train_rows+val_rows}
    validate_splits(list(unique.values()))
    if len({r['source_video_id'] for r in train_rows}&{r['source_video_id'] for r in val_rows}):
        raise ValueError('source video crosses train/validation')
    output=Path(output)
    if output.exists():
        raise FileExistsError('Gate2 training outputs are immutable')
    output.mkdir(parents=True)
    scale=max(float(np.quantile([abs(r['nll_utility']) for r in train_rows],.75)),1e-4)
    delta=float(np.quantile([abs(r['nll_utility']) for r in train_rows],.25))
    config={'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'train_targets_sha256':sha256(train_file),'val_targets_sha256':sha256(val_file),
        'training_data_role':'official AVQA train only','test_targets_read_during_fit':False,
        'mode':'measured_segment' if segment else 'prefix_feasibility_proxy','epochs':epochs,
        'seeds':[1234,2333,3407],'auxiliary_weights':[0.,.2],'weighting':['uniform','reliability'],
        'layouts':['anchor_history'] if segment else ['anchor_history','pooled_history'],
        'primary_layout':'anchor_history','primary_seed':1234,'target_scale':scale,'delta':delta,
        'target_statistics_fit_split':'train','configuration_and_threshold_fit_split':'val',
        'feature_layout':'current visual/audio + strictly past means, causal timing, availability',
        'modality_names':['audio'],'query_or_gold_input_to_gate':False,
        'feature_source_sha256':sha256(Path(__file__).with_name('gate2_interfaces.py'))}
    config.update(experiment_source_sha256=sha256(__file__),
        availability_scope='current window' if segment else 'any available candidate audio within legal prefix; routing still masks each window',
        target_audio_endpoint=train_rows[0].get('audio_endpoint','measured window'))
    write_json(output/'train_config.json',config)
    candidates=[];layout_cache={}
    for layout in config['layouts']:
        train_x,_=load_examples(train_rows,layout)
        val_x,val_energy=load_examples(val_rows,layout)
        layout_cache[layout]=(val_x,val_energy)
        for auxiliary in config['auxiliary_weights']:
            for weighting in config['weighting']:
                for seed in config['seeds']:
                    model,_,history,diagnostics=fit_one(train_rows,val_rows,train_x,val_x,scale,delta,
                        seed,auxiliary,weighting,epochs)
                    with torch.inference_mode():scores=model(val_x).utility_score.numpy()
                    threshold=stable_threshold(val_rows,scores,val_x.valid_audio.numpy())
                    name=f'{layout}_aux{auxiliary:g}_{weighting}_seed{seed}'
                    checkpoint=output/(name+'.pt')
                    torch.save({'kind':'visual_anchored_audio_gate','spec':model.spec,'state_dict':model.state_dict(),
                        'layout':layout,'mode':config['mode'],'train_config':config,'diagnostics':diagnostics,
                        'seed':seed,'auxiliary_weight':auxiliary,'weighting':weighting},checkpoint)
                    threshold.update(model_sha256=sha256(checkpoint))
                    write_json(output/(name+'.threshold.json'),threshold)
                    write_json(output/(name+'.history.json'),history)
                    candidates.append({'name':name,'layout':layout,'seed':seed,'auxiliary':auxiliary,'weighting':weighting,
                        'checkpoint':str(checkpoint.resolve()),'threshold_file':str((output/(name+'.threshold.json')).resolve()),
                        'validation':threshold['validation'],'diagnostics':diagnostics})
                    print(json.dumps({'trained':name,'validation_accuracy':threshold['validation']['accuracy'],
                        'validation_nll':threshold['validation']['gold_nll'],'parameters':diagnostics['parameters']}),flush=True)
    primary_by_layout={}
    for layout in config['layouts']:
        primary_by_layout[layout]=min([c for c in candidates if c['layout']==layout and c['seed']==1234],
            key=lambda r:(-r['validation']['accuracy'],r['validation']['gold_nll'],r['validation']['gate_on_rate'],r['name']))
    selected=[c for c in candidates if c['auxiliary']==primary_by_layout[c['layout']]['auxiliary']
              and c['weighting']==primary_by_layout[c['layout']]['weighting']]
    val_x,val_energy=layout_cache['anchor_history']
    energy=stable_threshold(val_rows,val_energy,val_x.valid_audio.numpy())
    lock={'primary':primary_by_layout['anchor_history'],'primary_by_layout':primary_by_layout,
        'replicates':selected,'candidates':candidates,'energy_threshold':energy,
        'test_used_to_select':False,'frozen_before_test_evaluation':True,
        'legal_training_and_validation_source':'AVQA official train_qa.json'}
    write_json(output/'selection.json',lock)
    evaluate_gate2(train_file,output,'train')
    evaluate_gate2(val_file,output,'val')
    return lock


def evaluate_gate2(targets,model_dir,split='test'):
    model_dir=Path(model_dir)
    lock=json.loads((model_dir/'selection.json').read_text())
    rows=read_jsonl(targets)
    if not rows or any(r['split']!=split for r in rows):
        raise ValueError('evaluation targets do not match requested split')
    results=[];cached={}
    for entry in lock['replicates']:
        layout=entry['layout']
        if layout not in cached:cached[layout]=load_examples(rows,layout)
        features,energy=cached[layout]
        gate,payload=load_gate2(entry['checkpoint'])
        threshold=json.loads(Path(entry['threshold_file']).read_text())
        if threshold['model_sha256']!=sha256(entry['checkpoint']) or threshold['fit_split']!='val':
            raise ValueError('frozen Gate2 checkpoint/threshold mismatch')
        prediction=gate.predict(features)
        scores=prediction.utility_score.numpy()
        accepted=(scores>threshold['threshold'])&prediction.valid_audio.numpy()
        metrics={'split':split,'method':layout,'seed':entry['seed'],**regression_metrics(rows,scores,accepted)}
        if split=='test':metrics['video_bootstrap_gain_over_audio']=video_bootstrap_gain(rows,accepted)
        results.append(metrics)
        write_jsonl(model_dir/f'{split}_{layout}_seed{entry["seed"]}.predictions.jsonl',[
            {'question_id':r['question_id'],'source_video_id':r['source_video_id'],'accepted':bool(a),
             'utility_score':float(s),'actual_accuracy_utility':r['accuracy_utility'],
             'actual_nll_utility':r['nll_utility'],'availability':bool(v)}
            for r,a,s,v in zip(rows,accepted,scores,prediction.valid_audio.numpy())])
        if layout!='anchor_history' or entry['seed']!=1234:continue
        valid=prediction.valid_audio.numpy()
        decisions={'always_off':np.zeros(len(rows),bool),'always_audio_endpoint':np.ones(len(rows),bool),
            'availability_only':valid,'energy_gate':(energy>lock['energy_threshold']['threshold'])&valid,
            'random_matched_rate':random_selector(rows,valid,int(accepted.sum())),
            'offline_endpoint_oracle':np.asarray([r['accuracy_utility']>0 or (r['accuracy_utility']==0 and r['nll_utility']>0) for r in rows])}
        for name,choice in decisions.items():results.append({'split':split,'method':name,'seed':1234,**selection_metrics(rows,choice)})
    write_json(model_dir/(split+'_metrics.json'),results)
    fields=['split','method','seed','n','accuracy','gold_nll','gate_on_rate','accuracy_gain_over_audio',
        'accuracy_gain_over_vision','false_allow_harmful','false_reject_helpful','MAE','Huber','Spearman']
    with (model_dir/(split+'_metrics.csv')).open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(results)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['labels','train','evaluate'])
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--paired',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--train-targets',type=Path)
    parser.add_argument('--val-targets',type=Path)
    parser.add_argument('--test-targets',type=Path)
    parser.add_argument('--segment',action='store_true')
    args=parser.parse_args()
    if args.stage=='labels':export_labels(args.manifest,args.paired,args.output)
    elif args.stage=='train':train_gate2(args.train_targets,args.val_targets,args.output,segment=args.segment)
    else:evaluate_gate2(args.test_targets,args.output)


if __name__=='__main__':main()
