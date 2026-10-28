"""Measured-window Gate2 supervision and chronological audio-only deployment."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path

import numpy as np
import torch

from .common import append_jsonl,fingerprint,read_jsonl,seed_all,sha256,write_json,write_jsonl
from .features import take_features
from .gate2_interfaces import AuxiliaryStream,assemble_anchor_features,causal_audio_pool,causal_audio_mask,load_gate2
from .gate2_runtime import qualified_latents
from .runtime import score_answer
from .strict_runtime import StrictStreamingRuntime


def plan_subset(manifest,output):
    rows=read_jsonl(manifest)
    selected=[]
    for split,count in [('train',32),('val',16),('test',64)]:
        candidates=[r for r in rows if r['split']==split]
        if split=='test':
            candidates=sorted(candidates,key=lambda r:({'View':0,'Sound':1,'Both':2}[r['question_modality']],
                                                       fingerprint([1234,'gate2-temporal-test',r['question_id']])))
        else:
            candidates=sorted(candidates,key=lambda r:fingerprint([1234,'gate2-temporal-development',r['question_id']]))
        selected.extend(candidates[:count])
    output=Path(output)
    if output.exists():
        if read_jsonl(output)!=selected:raise ValueError('frozen temporal subset changed')
    else:write_jsonl(output,selected)
    return selected


class Gate2OnlineState:
    """Only current observations enter step; old routed tokens are immutable."""
    def __init__(self,runtime,gate,threshold):
        self.runtime,self.gate,self.threshold=runtime,gate,float(threshold)
        self.visual_means,self.audio_means,self.valid,self.timestamps=[],[],[],[]
        self.latents,self.trace=[],[]

    @torch.inference_mode()
    def step(self,visual,audio,timestamp,available,source_audio_end=None,frame_pts=None,force=None):
        timestamp=float(timestamp)
        if self.timestamps and timestamp<=self.timestamps[-1]:raise ValueError('non-increasing stream clock')
        if frame_pts is not None and frame_pts>timestamp+1e-6:raise ValueError('future visual frame')
        source_end=timestamp if source_audio_end is None else float(source_audio_end)
        a=(audio if audio is not None else torch.zeros(1,768,device=visual.device))
        stream=AuxiliaryStream(a[None,None],torch.full((1,1,len(a)),source_end,device=visual.device),
            torch.tensor([[bool(available) and audio is not None]],device=visual.device),'audio')
        pooled,current_valid=causal_audio_pool(stream,torch.tensor([[timestamp]],device=visual.device))
        okay=bool(current_valid[0,0])
        causal_states=a[causal_audio_mask(stream,torch.tensor([[timestamp]],device=visual.device))[0,0]]
        self.visual_means.append(visual.detach().float().mean(0).clone())
        self.audio_means.append(pooled[0,0].clone())
        self.valid.append(okay)
        previous=self.timestamps[-1] if self.timestamps else None
        self.timestamps.append(timestamp)
        features=assemble_anchor_features(torch.stack(self.visual_means)[None],torch.stack(self.audio_means)[None],
            torch.tensor([self.valid],device=visual.device),torch.tensor([self.timestamps],device=visual.device))
        current=take_features(features,slice(-1,None))
        prediction=self.gate.predict(current)
        accepted=okay and bool(prediction.utility_score[0]>self.threshold)
        if force is not None:accepted=okay and bool(force)
        if accepted:
            # Original independent batch rows use the preceding timestamp for
            # delta-time conditioning. Duplicate CURRENT observations supply
            # that metadata; no historical visual frame is rerouted.
            if previous is None:
                z,p=self.runtime.audio_router(visual[None],causal_states[None],torch.tensor([timestamp],device=visual.device))
            else:
                z,p=self.runtime.audio_router(visual[None].expand(2,-1,-1),causal_states[None].expand(2,-1,-1),
                    torch.tensor([previous,timestamp],device=visual.device))
                z,p=z[-1:],p[-1:]
        else:
            z,p=self.runtime.vision_router(visual[None],None,torch.tensor([timestamp],device=visual.device))
        self.latents.append(z[0].clone())
        self.trace.append({'timestamp_s':timestamp,'frame_pts_s':frame_pts,'availability':okay,
            'source_audio_end_s':source_end,'source_audio_start_s':max(0.,source_end-2.),
            'gate':float(self.gate.last_output.gates[0,0,0]),'utility_score':float(prediction.utility_score[0]),
            'accepted':accepted,'threshold':self.threshold,'native_visual_slots':64,'historical_tokens_revised':False})
        return z[0],p[0]

    def bank(self):return torch.stack(self.latents)


def cached_stream(runtime,gate,threshold,payload,audio=None,available=None,source_ends=None,force=None):
    result=Gate2OnlineState(runtime,gate,threshold)
    audio=payload['causal_audio'] if audio is None else audio
    available=payload['audio_valid'] if available is None else available
    source_ends=payload['timestamps'].tolist() if source_ends is None else source_ends
    for index,time in enumerate(payload['timestamps'].tolist()):
        result.step(payload['visual'][index].to(runtime.device),audio[index].to(runtime.device),time,
            bool(available[index]),source_ends[index],payload['frame_pts_s'][index],None if force is None else force[index])
    return result


def generate_window_targets(manifest,paired,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    runtime=StrictStreamingRuntime();seed_all(1234)
    references={r['question_id']:r for r in read_jsonl(paired)}
    rows=[r for r in read_jsonl(manifest) if r['split'] in ('train','val')]
    path=output/'window_utility.jsonl'
    previous=read_jsonl(path) if path.exists() else []
    done={(r['question_id'],r['window_index']) for r in previous}
    for row in rows:
        if row['official_split']!='train':raise ValueError('evaluation QA cannot generate training window targets')
        ref=references[row['question_id']]
        payload=torch.load(ref['feature_cache_path'],map_location='cpu',weights_only=True)
        all_audio,_=qualified_latents(runtime,payload)
        all_audio=all_audio[0]
        vision=payload['vision_latents'].to(runtime.device)
        baseline=ref['branches']['audio']
        for index,time in enumerate(payload['timestamps'].tolist()):
            if (row['question_id'],index) in done:continue
            replaced=all_audio.clone();replaced[index]=vision[index]
            if torch.equal(replaced,all_audio):
                scored=baseline;reused=True
            else:
                scored=score_answer(runtime.generate(runtime.prompt(row),replaced),row['answer'],row['dataset']);reused=False
            label={k:ref[k] for k in ('question_id','source_video_id','videoID','video_path','media_sha256','dataset','split',
                'task_type','query_timestamp_s','question_modality','feature_cache_key','feature_cache_path','config_fingerprint')}
            label.update(window_index=index,window_timestamp_s=time,official_split='train',
                annotation_source_commit=row['annotation_source_commit'],
                audio_endpoint='availability-qualified original audio; single-window off counterfactual',
                audio_correct=baseline['correct'],vision_correct=scored['correct'],
                audio_gold_nll=baseline['gold_nll'],vision_gold_nll=scored['gold_nll'],
                nll_utility=scored['gold_nll']-baseline['gold_nll'],
                accuracy_utility=int(baseline['correct'])-int(scored['correct']),
                source_audio_end_s=time,source_audio_start_s=max(0.,time-2.),
                available_current_audio=bool(payload['audio_valid'][index]),
                label_scope='measured single-window marginal; no whole-clip label replication',
                identical_token_bank_reuses_recorded_generation=reused,
                original_gold_answer=row['answer'])
            append_jsonl(path,label)
        print(json.dumps({'credited':row['question_id'],'split':row['split'],'windows':len(all_audio)}),flush=True)
    labels=read_jsonl(path)
    for split in ('train','val'):
        write_jsonl(output/'training_assets'/(split+'.jsonl'),[r for r in labels if r['split']==split])
    write_json(output/'window_statistics.json',{'windows':len(labels),'by_split':dict(Counter(r['split'] for r in labels)),
        'correctness_deltas':dict(Counter(r['accuracy_utility'] for r in labels)),
        'test_window_targets_created':False,'source':'official AVQA train only',
        'limitation':'single-window marginal contributions can interact under simultaneous switching'})


def calibrate(manifest,paired,model_dir,output):
    output=Path(output)
    if output.exists():raise FileExistsError('stream threshold output is immutable')
    output.mkdir(parents=True)
    runtime=StrictStreamingRuntime();seed_all(1234)
    refs={r['question_id']:r for r in read_jsonl(paired)}
    rows=[r for r in read_jsonl(manifest) if r['split']=='val']
    selection=json.loads((Path(model_dir)/'selection.json').read_text());primary=selection['primary']
    gate,payload_gate=load_gate2(primary['checkpoint'],runtime.device)
    if payload_gate['mode']!='measured_segment':raise ValueError('strict streaming requires measured-window training')
    values=[]
    validation_available=0
    validation_windows=0
    for row in rows:
        p=torch.load(refs[row['question_id']]['feature_cache_path'],map_location='cpu',weights_only=True)
        values.extend(t['utility_score'] for t in cached_stream(runtime,gate,0.,p).trace if t['availability'])
        validation_available+=int(p['audio_valid'].sum())
        validation_windows+=len(p['audio_valid'])
    if not values:raise ValueError('no available validation audio')
    candidates=np.unique(np.concatenate(([min(values)-max(1.,max(np.abs(values)))],np.quantile(values,np.linspace(0,1,11)))))
    trials=[]
    for threshold in candidates:
        results=[]
        for row in rows:
            p=torch.load(refs[row['question_id']]['feature_cache_path'],map_location='cpu',weights_only=True)
            state=cached_stream(runtime,gate,float(threshold),p)
            scored=score_answer(runtime.generate(runtime.prompt(row),state.bank()),row['answer'],row['dataset'])
            results.append({'question_id':row['question_id'],'correct':scored['correct'],'gold_nll':scored['gold_nll'],
                'accepted':sum(t['accepted'] for t in state.trace),'windows':len(state.trace)})
        trial={'threshold':float(threshold),'accuracy':float(np.mean([r['correct'] for r in results])),
            'gold_nll':float(np.mean([r['gold_nll'] for r in results])),
            'gate_on_rate':sum(r['accepted'] for r in results)/sum(r['windows'] for r in results)}
        trials.append(trial);append_jsonl(output/'validation_trials.jsonl',{'trial':trial,'QA':results})
        print(json.dumps({'validated':trial}),flush=True)
    best=min(trials,key=lambda r:(-r['accuracy'],r['gold_nll'],r['gate_on_rate'],-r['threshold']))
    write_json(output/'streaming_threshold.json',{'fit_split':'val','threshold':best['threshold'],
        'validation':best,'curve':trials,'model_sha256':sha256(primary['checkpoint']),
        'test_QA_used':False,'model_checkpoint':primary['checkpoint'],
        'random_conditional_probability':min(1.,best['gate_on_rate']*validation_windows/max(1,validation_available)),
        'validation_available_window_fraction':validation_available/validation_windows})


def evaluate(manifest,paired,calibration,output):
    output=Path(output)
    if output.exists():raise FileExistsError('stream evaluation outputs are immutable')
    output.mkdir(parents=True)
    runtime=StrictStreamingRuntime();seed_all(1234)
    config=json.loads(Path(calibration).read_text())
    if config['fit_split']!='val' or sha256(config['model_checkpoint'])!=config['model_sha256']:
        raise ValueError('stream model/validation threshold mismatch')
    gate,payload_gate=load_gate2(config['model_checkpoint'],runtime.device)
    refs={r['question_id']:r for r in read_jsonl(paired)}
    rows=[r for r in read_jsonl(manifest) if r['split']=='test']
    results=[]
    for row in rows:
        ref=refs[row['question_id']]
        p=torch.load(ref['feature_cache_path'],map_location='cpu',weights_only=True)
        state=cached_stream(runtime,gate,config['threshold'],p)
        generation=runtime.generate(runtime.prompt(row),state.bank())
        scored=score_answer(generation,row['answer'],row['dataset'])
        # Validation-rate online Bernoulli control; only current availability is
        # applied. Its TEST rate is reported rather than forced with future masks.
        rng=np.random.default_rng(int(fingerprint([1234,'online-random',row['question_id']])[:8],16))
        force=rng.random(len(state.trace))<config['random_conditional_probability']
        random_state=cached_stream(runtime,gate,config['threshold'],p,force=force)
        random_score=score_answer(runtime.generate(runtime.prompt(row),random_state.bank()),row['answer'],row['dataset'])
        record={'question_id':row['question_id'],'source_video_id':row['source_video_id'],
            'question_modality':row['question_modality'],'task_type':row['task_type'],
            'correct':scored['correct'],'gold_nll':scored['gold_nll'],'generated_ids':generation['generated_ids'],
            'visual_tokens':generation['visual_tokens'],'trace':state.trace,
            'vision_correct':ref['branches']['vision']['correct'],'safe_audio_correct':ref['branches']['audio']['correct'],
            'legacy_audio_correct':ref['branches']['legacy_audio']['correct'],
            'full_vision_correct':ref['branches']['full_vision']['correct'],
            'random_correct':random_score['correct'],'random_trace':random_state.trace,
            'decision_inputs':'current visual/audio plus strictly past state, availability; no QA',
            'question_passed_to_VLM_only_after_routing':True,'test_window_oracles_created':False}
        append_jsonl(output/'streaming_results.jsonl',record);results.append(record)
        print(json.dumps({'streamed':row['question_id'],'correct':scored['correct']}),flush=True)
    windows=sum(len(r['trace']) for r in results)
    summary={'n':len(results),'accuracy':float(np.mean([r['correct'] for r in results])),
        'vision_accuracy':float(np.mean([r['vision_correct'] for r in results])),
        'safe_audio_accuracy':float(np.mean([r['safe_audio_correct'] for r in results])),
        'legacy_audio_accuracy':float(np.mean([r['legacy_audio_correct'] for r in results])),
        'full_vision_accuracy':float(np.mean([r['full_vision_correct'] for r in results])),
        'random_accuracy':float(np.mean([r['random_correct'] for r in results])),
        'gate_on_rate':sum(t['accepted'] for r in results for t in r['trace'])/windows,
        'random_on_rate':sum(t['accepted'] for r in results for t in r['random_trace'])/windows,
        'random_policy':'causal Bernoulli conditional on availability, fitted to validation on-rate; test rate reported',
        'total_windows':windows,'threshold_frozen':True,'test_frame_utility_targets_created':False,
        'historical_tokens_never_revised':True,'mode':'chronological cached-encoder replay; raw replay is a separate verification'}
    write_json(output/'summary.json',summary)


def robustness(manifest,paired,calibration,output):
    output=Path(output)
    if output.exists():raise FileExistsError('robustness outputs are immutable')
    output.mkdir(parents=True)
    runtime=StrictStreamingRuntime();seed_all(1234)
    config=json.loads(Path(calibration).read_text())
    if config['fit_split']!='val' or sha256(config['model_checkpoint'])!=config['model_sha256']:
        raise ValueError('robustness cannot refit the frozen streaming model')
    gate,_=load_gate2(config['model_checkpoint'],runtime.device)
    refs={r['question_id']:r for r in read_jsonl(paired)}
    rows=sorted([r for r in read_jsonl(manifest) if r['split']=='test'],key=lambda r:r['question_id'])
    results=[]
    conditions=('missing','silence','white_noise_0db','past_shift_2s','wrong_audio')
    for index,row in enumerate(rows):
        ref=refs[row['question_id']]
        p=torch.load(ref['feature_cache_path'],map_location='cpu',weights_only=True)
        for condition in conditions:
            windows=p['audio_windows'].clone();valid=p['audio_valid'].clone()
            ends=p['timestamps'].tolist();donor=None
            if condition in ('missing','silence'):
                windows.zero_();valid.zero_()
            elif condition=='white_noise_0db':
                rng=torch.Generator().manual_seed(int(fingerprint([1234,row['question_id'],condition])[:8],16))
                windows+=torch.randn(windows.shape,generator=rng)*windows.square().mean(1,keepdim=True).sqrt()
            elif condition=='past_shift_2s':
                ends=[max(0.,t-2.) for t in ends]
                windows,valid,_,_=runtime.causal_windows(row['video_path'],ends)
            else:
                donor=rows[(index+1)%len(rows)]
                if donor['source_video_id']==row['source_video_id']:raise ValueError('wrong-audio donor was not a different original video')
                windows,valid,_,_=runtime.causal_windows(donor['video_path'],ends)
            audio=runtime.audio_encoder.encode_windows(windows,runtime.device).float()
            learned=cached_stream(runtime,gate,config['threshold'],p,audio=audio,available=valid,source_ends=ends)
            always=cached_stream(runtime,gate,config['threshold'],p,audio=audio,available=valid,source_ends=ends,
                                 force=np.ones(len(ends),dtype=bool))
            prompt=runtime.prompt(row)
            g=score_answer(runtime.generate(prompt,learned.bank()),row['answer'],row['dataset'])
            a=score_answer(runtime.generate(prompt,always.bank()),row['answer'],row['dataset'])
            record={'question_id':row['question_id'],'source_video_id':row['source_video_id'],'condition':condition,
                'question_modality':row['question_modality'],'gated_correct':g['correct'],'safe_audio_correct':a['correct'],
                'vision_correct':ref['branches']['vision']['correct'],'gated_nll':g['gold_nll'],'safe_audio_nll':a['gold_nll'],
                'trace':learned.trace,'source_audio_ends_causal':all(t<=now+1e-8 for t,now in zip(ends,p['timestamps'].tolist())),
                'donor_source_video_id':donor['source_video_id'] if donor is not None else None,
                'donor_split':donor['split'] if donor is not None else None,
                'corruption_itself_never_used_as_a_utility_label':True,
                'interaction_harm_despite_both_endpoints_correct':bool(a['correct'] and ref['branches']['vision']['correct'] and not g['correct'])}
            append_jsonl(output/'robustness_results.jsonl',record);results.append(record)
        print(json.dumps({'robustness_clip':row['question_id'],'conditions':len(conditions)}),flush=True)
    summary={}
    for condition in conditions:
        values=[r for r in results if r['condition']==condition]
        windows=sum(len(r['trace']) for r in values)
        harmful=[r for r in values if r['vision_correct'] and not r['safe_audio_correct']]
        summary[condition]={'n':len(values),'gated_accuracy':float(np.mean([r['gated_correct'] for r in values])),
            'safe_audio_accuracy':float(np.mean([r['safe_audio_correct'] for r in values])),
            'vision_accuracy':float(np.mean([r['vision_correct'] for r in values])),
            'gate_on_rate':sum(t['accepted'] for r in values for t in r['trace'])/windows,
            'harmful_audio_cases':len(harmful),
            'harmful_cases_not_recovered':sum(not r['gated_correct'] for r in harmful),
            'both_endpoints_correct_hybrid_wrong':sum(r['interaction_harm_despite_both_endpoints_correct'] for r in values)}
    write_json(output/'summary.json',{'conditions':summary,'model_and_threshold_frozen':True,
        'future_audio_used':False,'human_annotations_required':False,
        'white_noise_is_synthetic_not_natural_environment_recordings':True,
        'test_window_oracle_labels_created':False})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['plan','labels','calibrate','evaluate','robustness'])
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--paired',type=Path)
    parser.add_argument('--model-dir',type=Path)
    parser.add_argument('--calibration',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.stage=='plan':plan_subset(args.manifest,args.output)
    elif args.stage=='labels':generate_window_targets(args.manifest,args.paired,args.output)
    elif args.stage=='calibrate':calibrate(args.manifest,args.paired,args.model_dir,args.output)
    elif args.stage=='evaluate':evaluate(args.manifest,args.paired,args.calibration,args.output)
    else:robustness(args.manifest,args.paired,args.calibration,args.output)


if __name__=='__main__':main()
