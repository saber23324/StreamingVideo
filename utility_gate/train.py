"""M2: cached-feature Huber regression; all large modules stay outside training."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from .common import fingerprint, read_jsonl, seed_all, sha256, validate_splits, write_json, write_jsonl
from .features import GateFeatures, load_feature_payload, take_features
from .metrics import fit_threshold, regression_metrics, selection_metrics, random_selector, video_bootstrap_gain
from .models import TinyUtilityGate


def load_features(rows, segment=False):
    values, energy = [], []
    for row in rows:
        payload = torch.load(row['feature_cache_path'],map_location='cpu',weights_only=True)
        if payload['cache_key'] != row['feature_cache_key'] or payload['split'] != row['split'] or payload['source_video_id'] != row['source_video_id']:
            raise ValueError('cache key, source or split mismatch')
        value = load_feature_payload(payload['features'])
        index = row['window_index'] if segment else len(value.visual)-1
        if segment:
            value = GateFeatures(value.visual,value.audio,value.context,payload['audio_valid'].bool())
        values.append(take_features(value,slice(index,index+1)))
        energy.append(float(torch.log1p(payload['audio_rms'][:index+1].mean())))
    result = GateFeatures(*(torch.cat([getattr(x,key) for x in values]) for key in ('visual','audio','context','valid_audio')))
    result.validate()
    return result, np.asarray(energy)


def make_weights(rows, features, mode, delta):
    if mode not in ('uniform','reliability'):
        raise ValueError('unknown weighting ablation')
    weights = torch.ones(len(rows))
    if mode == 'reliability':
        nll = torch.tensor([abs(r['nll_utility']) for r in rows])
        weights = .25+.75*(nll/max(delta,1e-6)).clamp(max=1.)
        weights[torch.tensor([r['accuracy_utility'] != 0 for r in rows])] = 1.
    groups = Counter(r['source_video_id'] for r in rows)
    weights *= torch.tensor([1/groups[r['source_video_id']] for r in rows])
    weights *= features.valid_audio.float()
    if weights.sum() == 0:
        raise ValueError('no valid audio training rows')
    return weights / weights.sum() * len(weights)


def objective(model, features, target, accuracy, weights, auxiliary):
    prediction = model(features)
    scale = model.spec['target_scale']
    regression = F.huber_loss(prediction.utility_score/scale,target/scale,reduction='none')
    regression = (regression*weights).sum()/weights.sum().clamp_min(1e-6)
    discordant = (accuracy != 0) & features.valid_audio
    classification = regression*0
    if discordant.any():
        binary = (accuracy[discordant] == 1).float()
        classification = F.binary_cross_entropy_with_logits(prediction.beneficial_logit[discordant],binary,
                                                             weight=weights[discordant],reduction='sum')/weights[discordant].sum().clamp_min(1e-6)
    return regression+auxiliary*classification


def fit_one(train_rows, val_rows, train_x, val_x, scale, delta, seed, auxiliary, weighting, epochs):
    seed_all(seed)
    model = TinyUtilityGate(train_x.visual.shape[1],train_x.audio.shape[1],target_scale=scale)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01)
    train_y = torch.tensor([r['nll_utility'] for r in train_rows],dtype=torch.float32)
    train_c = torch.tensor([r['accuracy_utility'] for r in train_rows],dtype=torch.float32)
    val_y = torch.tensor([r['nll_utility'] for r in val_rows],dtype=torch.float32)
    val_c = torch.tensor([r['accuracy_utility'] for r in val_rows],dtype=torch.float32)
    weights = make_weights(train_rows,train_x,weighting,delta)
    val_weights = make_weights(val_rows,val_x,'uniform',delta)
    generator = torch.Generator().manual_seed(seed)
    best, history = None, []
    initial = {k:v.detach().clone() for k,v in model.state_dict().items()}
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(train_rows),generator=generator)
        losses = []
        for batch in permutation.split(32):
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model,take_features(train_x,batch),train_y[batch],train_c[batch],weights[batch],auxiliary)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite gate loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            prediction = model(val_x).utility_score
            valid_loss = float(F.huber_loss(prediction/scale,val_y/scale))
            val_mae = float((prediction-val_y).abs().mean())
            train_mae = float((model(train_x).utility_score-train_y).abs().mean())
        history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),
                        'val_huber_scaled':valid_loss,'val_MAE':val_mae,'train_MAE':train_mae})
        if best is None or valid_loss < best['loss']:
            best = {'loss':valid_loss,'epoch':epoch+1,'state':copy.deepcopy(model.state_dict())}
    model.load_state_dict(best['state'])
    model.eval()
    with torch.inference_mode():
        scores = model(val_x).utility_score.numpy()
    calibration = fit_threshold(val_rows,scores,val_x.valid_audio.numpy())
    changed = [name for name,value in model.state_dict().items() if not torch.equal(value,initial[name])]
    if not changed:
        raise AssertionError('gate parameters did not update')
    return model, calibration, history, {'best_epoch':best['epoch'],'seconds':time.perf_counter()-started,
        'parameters':sum(p.numel() for p in model.parameters()),'changed_gate_parameters':changed,
        'large_models_loaded_in_training':False,'all_input_features_detached':all(not t.requires_grad for t in
            (train_x.visual,train_x.audio,train_x.context,val_x.visual,val_x.audio,val_x.context))}


def train(labels_path, output, epochs=60, segment=False, seeds=(1234,2333,3407)):
    rows = read_jsonl(labels_path)
    if not segment:
        validate_splits(rows)
    else:
        # Multiple window records share a QA; audit unique QA/source identities.
        unique = {(r['question_id'],r['split']):r for r in rows}
        validate_splits(list(unique.values()))
    train_rows = [r for r in rows if r['split'] == 'train']
    val_rows = [r for r in rows if r['split'] == 'val']
    if not train_rows or not val_rows:
        raise ValueError('train and val are both required')
    output = Path(output)
    if output.exists():
        raise FileExistsError('training output is immutable; select a new directory')
    output.mkdir(parents=True)
    train_x,_ = load_features(train_rows,segment)
    val_x,val_energy = load_features(val_rows,segment)
    scale = max(float(np.quantile([abs(r['nll_utility']) for r in train_rows],.75)),1e-4)
    delta = float(np.quantile([abs(r['nll_utility']) for r in train_rows],.25))
    config = {'kind':'tiny_utility_mlp','mode':'causal_segment' if segment else 'prefix_proxy',
        'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'labels_sha256':sha256(labels_path),'seeds':list(seeds),'epochs':epochs,'batch_size':32,
        'learning_rate':.001,'weight_decay':.01,'target_scale':scale,'delta':delta,
        'scale_and_delta_fit_split':'train','threshold_fit_split':'val','primary_seed':1234,
        'candidate_auxiliary_weights':[0.,.2],'candidate_weighting':['uniform','reliability'],
        'checkpoint_selection':'minimum validation Huber; primary configuration by validation QA then NLL',
        'validation_class_counts':dict(Counter(r['accuracy_utility'] for r in val_rows)),
        'supervision':'measured per-window replacement' if segment else 'whole-prefix endpoint difference',
        'metric_branch_semantics':{'audio':'keep all-AV prefix','vision':'replace only this window'} if segment else
                                  {'audio':'all-AV prefix','vision':'all-vision prefix'},
        'inference_input_fields':['visual','audio','context','valid_audio']}
    write_json(output/'train_config.json',config)
    candidates = []
    for auxiliary in config['candidate_auxiliary_weights']:
        for weighting in config['candidate_weighting']:
            for seed in seeds:
                name = f'aux{auxiliary:g}_{weighting}_seed{seed}'
                model,calibration,history,diagnostics = fit_one(train_rows,val_rows,train_x,val_x,scale,delta,
                                                               seed,auxiliary,weighting,epochs)
                checkpoint = output/(name+'.pt')
                first_cache = torch.load(train_rows[0]['feature_cache_path'],map_location='cpu',weights_only=True)
                torch.save({'kind':'tiny_utility_mlp','spec':model.spec,'state_dict':model.state_dict(),
                    'mode':config['mode'],'train_config':config,'paired_config_fingerprint':first_cache['config_fingerprint'],
                    'seed':seed,'auxiliary_weight':auxiliary,'weighting':weighting,
                    'diagnostics':diagnostics},checkpoint)
                calibration.update(model_sha256=sha256(checkpoint),labels_sha256=sha256(labels_path))
                write_json(output/(name+'.threshold.json'),calibration)
                write_json(output/(name+'.history.json'),history)
                candidate = {'name':name,'seed':seed,'auxiliary_weight':auxiliary,'weighting':weighting,
                    'checkpoint':str(checkpoint.resolve()),'threshold_path':str((output/(name+'.threshold.json')).resolve()),
                    'validation':calibration['validation'],'diagnostics':diagnostics}
                candidates.append(candidate)
                print(json.dumps({'trained':name,'val_accuracy':calibration['validation']['accuracy'],
                                  'val_gold_nll':calibration['validation']['gold_nll'], 'seconds':diagnostics['seconds']}),flush=True)
    primary = min([c for c in candidates if c['seed'] == 1234],key=lambda c:
                  (-c['validation']['accuracy'],c['validation']['gold_nll'],c['validation']['gate_on_rate'],c['name']))
    selected = [c for c in candidates if c['auxiliary_weight'] == primary['auxiliary_weight'] and c['weighting'] == primary['weighting']]
    energy = fit_threshold(val_rows,val_energy,val_x.valid_audio.numpy())
    write_json(output/'energy_threshold.json',energy)
    lock = {'primary':primary,'replicate_seeds_same_configuration':selected,'all_candidates':candidates,
        'energy_threshold':energy,'selection_uses_test':False,
        'configuration_selection_seed':1234,'warning':'few validation discordant QA; threshold and configuration may overfit'}
    write_json(output/'selection.json',lock)
    evaluate(labels_path,output,segment)
    return lock


def evaluate(labels_path, output, segment=False):
    from .models import load_gate
    output = Path(output)
    lock = json.loads((output/'selection.json').read_text())
    all_rows = read_jsonl(labels_path)
    results = []
    for split in ('train','val','test'):
        rows = [r for r in all_rows if r['split'] == split]
        if not rows:
            continue
        features,energy = load_features(rows,segment)
        valid = features.valid_audio.numpy()
        for selected in lock['replicate_seeds_same_configuration']:
            model,payload = load_gate(selected['checkpoint'])
            threshold = json.loads(Path(selected['threshold_path']).read_text())
            if threshold['model_sha256'] != sha256(selected['checkpoint']):
                raise ValueError('checkpoint and frozen threshold mismatch')
            with torch.inference_mode():
                prediction = model(features)
            scores = prediction.utility_score.numpy()
            accepted = (scores > threshold['threshold']) & valid
            result = {'split':split,'seed':selected['seed'],'method':'tiny_utility_mlp',
                      **regression_metrics(rows,scores,accepted)}
            if split == 'test':
                result['video_bootstrap_gain_over_audio'] = video_bootstrap_gain(rows,accepted)
            results.append(result)
            write_jsonl(output/f'{split}_seed{selected["seed"]}.predictions.jsonl',
                [{'question_id':r['question_id'],'source_video_id':r['source_video_id'],
                  **({'window_index':r['window_index']} if segment else {}),
                  'utility_score':float(s),'accepted':bool(a),'valid_audio':bool(v),
                  'actual_accuracy_utility':r['accuracy_utility'],'actual_nll_utility':r['nll_utility']}
                 for r,s,a,v in zip(rows,scores,accepted,valid)])
            if selected['seed'] != 1234:
                continue
            selectors = {'always_vision':np.zeros(len(rows),bool),'always_audio':np.ones(len(rows),bool),
                'quality_only':valid,'energy_gate':(energy > lock['energy_threshold']['threshold']) & valid,
                'random_matched_on_rate':random_selector(rows,valid,int(accepted.sum())),
                'offline_oracle':np.asarray([r['accuracy_utility'] > 0 or (r['accuracy_utility'] == 0 and r['nll_utility'] > 0) for r in rows])}
            for name,decision in selectors.items():
                results.append({'split':split,'seed':1234,'method':name,**selection_metrics(rows,decision)})
    write_json(output/'gate_metrics.json',results)
    columns=['split','seed','method','n','accuracy','gold_nll','gate_on_rate','accuracy_gain_over_audio',
             'accuracy_gain_over_vision','MAE','Huber','Spearman','false_allow_harmful','false_reject_helpful']
    with (output/'gate_metrics.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(results)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=60)
    parser.add_argument('--segment',action='store_true')
    args=parser.parse_args()
    train(args.labels,args.output,args.epochs,args.segment)


if __name__ == '__main__':
    main()
