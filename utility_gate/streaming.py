"""M4 measured segment credit, validation calibration, and causal deployment."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from .common import (REPO, append_jsonl, fingerprint, read_jsonl, seed_all, sha256,
                     write_json, write_jsonl)
from .integration import policy_from_selection, verify_disabled
from .policy import CausalStreamRouter, UtilityRoutingPolicy
from .runtime import clocked, score_answer, synchronize
from .strict_runtime import StrictStreamingRuntime as FrozenRuntime


def replace_one_window(all_audio,vision,index):
    if all_audio.ndim!=3 or vision.shape!=all_audio.shape or not 0<=index<len(all_audio):
        raise ValueError('matching frame/slot/value tensors and a valid window index required')
    result=all_audio.clone()
    result[index]=vision[index]
    return result


def window_on_rates(records):
    total=sum(len(r['trace']) for r in records)
    accepted=sum(t['accepted'] for r in records for t in r['trace'])
    return {'total_windows':total,'accepted_windows':accepted,
            'gate_on_rate':accepted/total if total else 0.,
            'mean_video_gate_on_rate':float(np.mean([r['gate_on_rate'] for r in records])) if records else 0.}


def load_paired(manifest,paired,runtime):
    config=json.loads((Path(paired).parent/'paired_config.json').read_text())
    if config['mode']!='streaming' or config['runtime']!=runtime.metadata:
        raise ValueError('strict streaming endpoint model/encoder/clock contract mismatch')
    rows=read_jsonl(manifest)
    records={r['question_id']:r for r in read_jsonl(paired)}
    if set(records)!={r['question_id'] for r in rows}:
        raise ValueError('streaming paired data incomplete')
    return rows,records


def window_labels(manifest,paired,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    seed_all(1234)
    runtime=FrozenRuntime()
    rows,records=load_paired(manifest,paired,runtime)
    # Test answers are never used for window-label generation.
    rows=[r for r in rows if r['split'] in ('train','val')]
    path=output/'window_utility.jsonl'
    previous=read_jsonl(path) if path.exists() else []
    done={(r['question_id'],r['window_index']) for r in previous}
    if len(done)!=len(previous):
        raise ValueError('duplicate window credit records')
    for row in rows:
        reference=records[row['question_id']]
        payload=torch.load(reference['feature_cache_path'],map_location='cpu',weights_only=True)
        all_av=payload['audio_latents'].to(runtime.device)
        vision=payload['vision_latents'].to(runtime.device)
        baseline=reference['branches']['audio']
        prompt=runtime.prompt(row)
        for window in range(len(all_av)):
            if (row['question_id'],window) in done:
                continue
            replaced=replace_one_window(all_av,vision,window)
            scored=score_answer(runtime.generate(prompt,replaced),row['answer'],row['dataset'])
            label={k:reference[k] for k in ('question_id','source_video_id','videoID','video_path','media_sha256',
                'dataset','split','task_type','query_timestamp_s','question_modality','feature_cache_key',
                'feature_cache_path','config_fingerprint')}
            label.update(window_index=window,window_timestamp_s=float(payload['timestamps'][window]),
                source_audio_start_s=max(0.,float(payload['timestamps'][window])-2.),
                source_audio_end_s=float(payload['timestamps'][window]),
                audio_correct=baseline['correct'],vision_correct=scored['correct'],
                audio_gold_nll=baseline['gold_nll'],vision_gold_nll=scored['gold_nll'],
                nll_utility=scored['gold_nll']-baseline['gold_nll'],
                window_utility=scored['gold_nll']-baseline['gold_nll'],
                accuracy_utility=int(baseline['correct'])-int(scored['correct']),
                original_gold_answer=row['answer'],all_audio_generated_ids=baseline['generated_ids'],
                replace_window_generated_ids=scored['generated_ids'],
                valid_current_audio=bool(payload['audio_valid'][window]),
                supervision_scope='measured marginal replacement at this window; other windows remain AV',
                comparison_semantics={'audio':'all-AV prefix','vision':'only this window replaced with vision'},
                label_provenance='train/val QA gold; no future input to the segment gate')
            append_jsonl(path,label)
        print(json.dumps({'credited':row['question_id'],'split':row['split'],'windows':len(all_av)}),flush=True)
    labels=read_jsonl(path)
    if any(r['split']=='test' for r in labels):
        raise AssertionError('held-out test window labels are forbidden')
    write_json(output/'window_label_statistics.json',{'windows':len(labels),
        'by_split':dict(Counter(r['split'] for r in labels)),
        'questions':len({r['question_id'] for r in labels}),
        'source_videos':len({r['source_video_id'] for r in labels}),
        'correctness_deltas':dict(Counter(r['accuracy_utility'] for r in labels)),
        'nonfinite_targets':sum(not np.isfinite(r['nll_utility']) for r in labels),
        'test_window_labels_generated':False,'whole_prefix_labels_copied_to_frames':False,
        'interaction_limit':'single-window marginal effects need not add when several windows switch'})


def cached_bank(runtime,policy,payload):
    stream=CausalStreamRouter(policy)
    visual=payload['visual'].to(runtime.device)
    audio=payload['causal_audio'].to(runtime.device)
    for index,timestamp in enumerate(payload['timestamps'].tolist()):
        stream.step(visual[index],audio[index],timestamp,bool(payload['audio_valid'][index]),payload['frame_pts_s'][index])
    return stream


def calibrate_streaming(manifest,paired,selection_dir,output):
    output=Path(output)
    if output.exists():
        raise FileExistsError('strict validation threshold is frozen; use a fresh calibration directory')
    output.mkdir(parents=True)
    seed_all(1234)
    runtime=FrozenRuntime()
    rows,records=load_paired(manifest,paired,runtime)
    rows=[r for r in rows if r['split']=='val']
    policy,selection=policy_from_selection(runtime,selection_dir,'causal_segment')
    observed=[]
    for row in rows:
        payload=torch.load(records[row['question_id']]['feature_cache_path'],map_location='cpu',weights_only=True)
        trace=cached_bank(runtime,policy,payload).trace
        observed.extend(r['utility_score'] for r in trace if r['valid_audio'])
    if not observed:
        raise ValueError('no valid validation windows')
    thresholds=np.unique(np.concatenate(([min(observed)-max(1.,max(np.abs(observed)))],np.quantile(observed,np.linspace(0,1,11)))))
    trials=[]
    for threshold in thresholds:
        policy.threshold=float(threshold)
        results=[]
        for row in rows:
            payload=torch.load(records[row['question_id']]['feature_cache_path'],map_location='cpu',weights_only=True)
            stream=cached_bank(runtime,policy,payload)
            generation=runtime.generate(runtime.prompt(row),stream.token_bank())
            scored=score_answer(generation,row['answer'],row['dataset'])
            result={'question_id':row['question_id'],'threshold':float(threshold),'correct':scored['correct'],
                    'gold_nll':scored['gold_nll'],'accepted_windows':sum(r['accepted'] for r in stream.trace),
                    'windows':len(stream.trace),'generated_ids':generation['generated_ids']}
            results.append(result)
            append_jsonl(output/'validation_trials.jsonl',result)
        trial={'threshold':float(threshold),'accuracy':float(np.mean([r['correct'] for r in results])),
               'gold_nll':float(np.mean([r['gold_nll'] for r in results])),
               'gate_on_rate':sum(r['accepted_windows'] for r in results)/sum(r['windows'] for r in results)}
        trials.append(trial)
        print(json.dumps({'streaming_validation':trial}),flush=True)
    best=min(trials,key=lambda r:(-r['accuracy'],r['gold_nll'],r['gate_on_rate'],-r['threshold']))
    write_json(output/'streaming_threshold.json',{'fit_split':'val','comparison':'strict_greater_than',
        'threshold':best['threshold'],'validation':best,'trials':trials,
        'model_sha256':sha256(selection['primary']['checkpoint']),
        'window_model_selection_sha256':sha256(Path(selection_dir)/'selection.json'),
        'calibration_objective':'actual chronological mixed-window QA accuracy, then NLL, then lower on-rate',
        'test_answers_used':False,'candidates':'eleven validation-score quantiles plus all-eligible endpoint'})


def deploy_streaming(manifest,paired,selection_dir,calibration,output):
    """Read one arrival at a time from raw media; the question is used only at end."""
    from relevance_gate.media import frame_at_or_before
    output=Path(output)
    if output.exists():
        raise FileExistsError('use a fresh streaming test output')
    output.mkdir(parents=True)
    seed_all(1234)
    runtime=FrozenRuntime()
    rows,records=load_paired(manifest,paired,runtime)
    policy,selection=policy_from_selection(runtime,selection_dir,'causal_segment')
    config=json.loads(Path(calibration).read_text())
    if config['fit_split']!='val' or config['model_sha256']!=sha256(selection['primary']['checkpoint']):
        raise ValueError('strict streaming calibration mismatch')
    policy.threshold=config['threshold']
    write_json(output/'regression.json',verify_disabled(runtime))
    results=[]
    with torch.inference_mode():
        for row in [r for r in rows if r['split']=='test']:
            payload=torch.load(records[row['question_id']]['feature_cache_path'],map_location='cpu',weights_only=True)
            expected=cached_bank(runtime,policy,payload)
            stream=CausalStreamRouter(policy)
            runtime.strict_audio.clear()
            timing=Counter()
            started=time.perf_counter()
            interval=float(row.get('frame_interval_s',2.))
            for timestamp in np.arange(interval,row['query_time_seconds']+1e-8,interval):
                (image,pts),elapsed=clocked(lambda:frame_at_or_before(row['video_path'],float(timestamp)))
                timing['video_decode_s']+=elapsed
                visual,elapsed=clocked(lambda:runtime.extractor.encode_visual(runtime.model,runtime.processor,image[None],1).float()[0])
                timing['visual_encoder_s']+=elapsed
                (windows,valid,reasons,rms),elapsed=clocked(lambda:runtime.causal_windows(row['video_path'],[timestamp]))
                timing['audio_decode_s']+=elapsed
                if bool(valid[0]):
                    audio,elapsed=clocked(lambda:runtime.audio_encoder.encode_windows(windows,runtime.device).to(runtime.device).float()[0])
                    timing['audio_encoder_s']+=elapsed
                else:
                    audio=None
                previous_bank=stream.token_bank().clone() if stream.latents else None
                _,elapsed=clocked(lambda:stream.step(visual,audio,float(timestamp),bool(valid[0]),pts))
                timing['gate_and_routing_s']+=elapsed
                if previous_bank is not None and not torch.equal(previous_bank,stream.token_bank()[:-1]):
                    raise AssertionError('later observation revised historical routed tokens')
                stream.trace[-1].update(quality_reason=reasons[0],source_audio_end_s=float(timestamp),
                                       source_audio_start_s=max(0.,float(timestamp)-2.))
            # No question, option or answer was passed to any prior step.
            prompt,tokenization_s=clocked(lambda:runtime.prompt(row))
            generation=runtime.generate(prompt,stream.token_bank())
            processing=time.perf_counter()-started
            scored=score_answer(generation,row['answer'],row['dataset'])
            cached_generation=runtime.generate(runtime.prompt(row),expected.token_bank())
            record={'question_id':row['question_id'],'source_video_id':row['source_video_id'],'task_type':row['task_type'],
                'correct':scored['correct'],'gold_nll':scored['gold_nll'],'prediction':scored['prediction'],
                'generated_ids':generation['generated_ids'],'visual_tokens':generation['visual_tokens'],
                'gate_on_rate':sum(t['accepted'] for t in stream.trace)/len(stream.trace),
                'trace':stream.trace,'single_vlm_generation_for_deployed_policy':True,
                'cached_equivalence_generation_is_separate_test_only':True,
                'raw_vs_cached_generated_tokens_equal':generation['generated_ids']==cached_generation['generated_ids'],
                'raw_vs_cached_decisions_equal':[r['accepted'] for r in stream.trace]==[r['accepted'] for r in expected.trace],
                'historical_tokens_never_revised':True,'threshold':policy.threshold,
                'question_ttft_s':tokenization_s+generation['llm_ttft_s'],
                'question_tokenization_s':tokenization_s,'llm_ttft_s':generation['llm_ttft_s'],
                'llm_prefill_s':generation['llm_prefill_s'],
                'llm_total_s':generation['llm_total_s'],'processing_s_excluding_arrival_wait':processing,
                'observed_prefix_s':row['query_time_seconds'],'timing':dict(timing),
                'all_audio_encoding_included':True,'future_question_used_in_gate':False}
            append_jsonl(output/'streaming_gate_results.jsonl',record)
            results.append(record)
            print(json.dumps({'streamed':row['question_id'],'correct':scored['correct'],'on_rate':record['gate_on_rate']}),flush=True)
    summary={'questions':len(results),'accuracy':float(np.mean([r['correct'] for r in results])),
        **window_on_rates(results),
        'threshold':policy.threshold,'raw_vs_cached_generations_all_equal':all(r['raw_vs_cached_generated_tokens_equal'] for r in results),
        'raw_vs_cached_decisions_all_equal':all(r['raw_vs_cached_decisions_equal'] for r in results),
        'no_historical_recompression':True,'gate_inputs_causal':True,'question_agnostic_gate':True,
        'per_window_labels_measured_on_train_val_only':True,'test_window_oracle_generated':False,
        'scope':'bounded 2-second-arrival-grid streaming simulation; one VLM QA at question time; no wall-clock arrival sleeping',
        'timing_scope':'raw per-arrival decode/encoders/gate/routing and one VLM generation; excludes loading and real arrival wait',
        'audio_cost_included':True}
    write_json(output/'summary.json',summary)


def robustness(manifest,paired,selection_dir,calibration,output):
    output=Path(output)
    if output.exists():
        raise FileExistsError('fresh robustness output required')
    output.mkdir(parents=True)
    seed_all(1234)
    runtime=FrozenRuntime()
    rows,records=load_paired(manifest,paired,runtime)
    policy,selection=policy_from_selection(runtime,selection_dir,'causal_segment')
    threshold=json.loads(Path(calibration).read_text())
    if threshold['model_sha256']!=sha256(selection['primary']['checkpoint']) or threshold['fit_split']!='val':
        raise ValueError('robustness must use the frozen streaming calibration')
    policy.threshold=threshold['threshold']
    results=[]
    with torch.inference_mode():
        for row in [r for r in rows if r['split']=='test']:
            reference=records[row['question_id']]
            payload=torch.load(reference['feature_cache_path'],map_location='cpu',weights_only=True)
            visual=payload['visual'].to(runtime.device)
            timestamps=payload['timestamps'].tolist()
            for condition in ('silence','white_noise_0db','past_shift_2s'):
                windows=payload['audio_windows'].clone()
                valid=payload['audio_valid'].clone()
                source_ends=timestamps
                if condition=='silence':
                    windows.zero_();valid.zero_()
                elif condition=='white_noise_0db':
                    rng=torch.Generator().manual_seed(int(fingerprint([1234,row['question_id'],condition])[:8],16))
                    noise=torch.randn(windows.shape,generator=rng)
                    energy=windows.square().mean(1,keepdim=True).sqrt()
                    windows+=noise*energy
                else:
                    source_ends=[max(0.,t-2.) for t in timestamps]
                    windows,valid,_,_=runtime.causal_windows(row['video_path'],source_ends)
                audio=runtime.audio_encoder.encode_windows(windows,runtime.device).to(runtime.device).float()
                stream=CausalStreamRouter(policy)
                for index,timestamp in enumerate(timestamps):
                    stream.step(visual[index],audio[index],timestamp,bool(valid[index]),payload['frame_pts_s'][index])
                    stream.trace[-1].update(source_audio_end_s=source_ends[index],
                        source_audio_start_s=max(0.,source_ends[index]-2.),condition=condition)
                all_av,_=runtime.audio_router(visual,audio,payload['timestamps'].to(runtime.device))
                prompt=runtime.prompt(row)
                gated=score_answer(runtime.generate(prompt,stream.token_bank()),row['answer'],row['dataset'])
                always=score_answer(runtime.generate(prompt,all_av),row['answer'],row['dataset'])
                record={'question_id':row['question_id'],'source_video_id':row['source_video_id'],'condition':condition,
                    'gated_correct':gated['correct'],'always_audio_correct':always['correct'],
                    'vision_correct':reference['branches']['vision']['correct'],
                    'gated_gold_nll':gated['gold_nll'],'always_audio_gold_nll':always['gold_nll'],
                    'gate_on_rate':sum(r['accepted'] for r in stream.trace)/len(stream.trace),
                    'trace':stream.trace,'all_source_audio_ends_causal':all(t<=now+1e-8 for t,now in zip(source_ends,timestamps)),
                    'corruption_not_assigned_utility_label':True}
                append_jsonl(output/'robustness_results.jsonl',record);results.append(record)
    summary={}
    for condition in ('silence','white_noise_0db','past_shift_2s'):
        selected=[r for r in results if r['condition']==condition]
        summary[condition]={'n':len(selected),'gated_accuracy':float(np.mean([r['gated_correct'] for r in selected])),
            'always_audio_accuracy':float(np.mean([r['always_audio_correct'] for r in selected])),
            'vision_accuracy':float(np.mean([r['vision_correct'] for r in selected])),
            **window_on_rates(selected)}
    write_json(output/'summary.json',{'conditions':summary,'threshold_frozen':True,
        'future_audio_used':False,'semantic_negative_labels_created':False,
        'not_run':['real background-music mixing','real environmental-noise recordings'],
        'white_noise_note':'synthetic Gaussian 0dB perturbation, not a natural environmental-noise dataset'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['labels','calibrate','deploy','robustness'])
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--paired',type=Path,required=True)
    parser.add_argument('--selection-dir',type=Path)
    parser.add_argument('--calibration',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.stage=='labels':
        window_labels(args.manifest,args.paired,args.output)
    elif args.stage=='calibrate':
        calibrate_streaming(args.manifest,args.paired,args.selection_dir,args.output)
    elif args.stage=='deploy':
        deploy_streaming(args.manifest,args.paired,args.selection_dir,args.calibration,args.output)
    else:
        robustness(args.manifest,args.paired,args.selection_dir,args.calibration,args.output)


if __name__=='__main__':
    main()
