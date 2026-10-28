"""Availability-qualified endpoints and exact Gate2 frozen-VLM integration."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from .common import REPO,append_jsonl,fingerprint,read_jsonl,seed_all,sha256,write_json,write_jsonl
from .features import GateFeatures,take_features
from .gate2_interfaces import (AuxiliaryStream,SelectedAudioQueryRouter,features_from_cache,load_gate2)
from .runtime import score_answer
from .strict_runtime import StrictStreamingRuntime


def stream_from_payload(payload,device):
    audio=payload['causal_audio'][None].to(device)
    time=payload['timestamps'][None].to(device).float()
    return AuxiliaryStream(audio,time[:,:,None].expand(audio.shape[:-1]),payload['audio_valid'][None].to(device).bool(),'audio')


def qualified_latents(runtime,payload):
    router=SelectedAudioQueryRouter(runtime.audio_router,runtime.vision_router)
    visual=payload['visual'][None].to(runtime.device)
    timestamps=payload['timestamps'][None].to(runtime.device)
    return router(visual,stream_from_payload(payload,runtime.device),timestamps,
                  torch.ones(timestamps.shape,device=runtime.device,dtype=torch.bool))


def qualify_endpoints(manifest,paired,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=read_jsonl(manifest);original={r['question_id']:r for r in read_jsonl(paired)}
    if set(original)!={r['question_id'] for r in rows}:
        raise ValueError('raw paired QA incomplete')
    seed_all(1234)
    runtime=StrictStreamingRuntime()
    parent=json.loads((Path(paired).parent/'paired_config.json').read_text())
    if parent['runtime']!=runtime.metadata:
        raise ValueError('encoder/scorer changed after raw paired evaluation')
    config={**parent,'parent_paired_sha256':sha256(paired),
        'audio_branch':'same original audio checkpoint; invalid windows explicitly use original vision checkpoint',
        'legacy_audio_branch_retained':True,
        'selected_query_router_source_sha256':sha256(Path(__file__).with_name('gate2_interfaces.py'))}
    config['fingerprint']=fingerprint(config)
    write_json(output/'paired_config.json',config)
    path=output/'paired_qa.jsonl'
    previous=read_jsonl(path) if path.exists() else []
    done={r['question_id'] for r in previous}
    if len(done)!=len(previous) or any(r['config_fingerprint']!=config['fingerprint'] for r in previous):
        raise ValueError('qualified endpoint resume mismatch')
    reuse={'all_valid':0,'all_invalid':0,'mixed':0}
    for row in rows:
        if row['question_id'] in done:continue
        reference=original[row['question_id']]
        payload=torch.load(reference['feature_cache_path'],map_location='cpu',weights_only=True)
        valid=payload['audio_valid'].bool()
        z,p=qualified_latents(runtime,payload)
        if bool(valid.all()):
            if not torch.equal(z[0].cpu(),payload['audio_latents']):
                raise AssertionError('qualified all-valid endpoint differs from original AudioRouter')
            scored=reference['branches']['audio'];kind='all_valid'
        elif not bool(valid.any()):
            if not torch.equal(z[0].cpu(),payload['vision_latents']):
                raise AssertionError('all-unavailable endpoint differs from the original vision fallback')
            scored=reference['branches']['vision'];kind='all_invalid'
        else:
            scored=score_answer(runtime.generate(runtime.prompt(row),z[0]),row['answer'],row['dataset']);kind='mixed'
        result=copy.deepcopy(reference)
        result['branches']['legacy_audio']=reference['branches']['audio']
        result['branches']['audio']=scored
        result.update(config_fingerprint=config['fingerprint'],
            accuracy_utility=int(scored['correct'])-int(reference['branches']['vision']['correct']),
            nll_utility=reference['branches']['vision']['gold_nll']-scored['gold_nll'],
            audio_endpoint_validity=kind,encoder_cache_reused_without_mutation=True,
            audio_endpoint='availability-qualified original AudioRouter/vision windows',
            legacy_endpoint_generation_reused=kind!='mixed')
        reuse[kind]+=1
        append_jsonl(path,result)
    final=read_jsonl(path)
    if len(final)!=len(rows):raise ValueError('qualified paired set incomplete')
    write_json(output/'qualification.json',{'questions':len(final),'counts':dict(__import__('collections').Counter(r['audio_endpoint_validity'] for r in final)),
        'new_VLM_generations_required':'mixed-validity windows only',
        'identical_all-valid/all-unavailable_token_inputs_reuse_recorded_generation':True,
        'no_encoder_cache_mutated':True,'same_native64_slots_per_window':True})
    print(json.dumps({'qualified':len(final),'counts':reuse}),flush=True)


def verify_protocol(runtime):
    fixture=torch.load(REPO/'artifacts/relevance_gate/m0-gate-disabled-final-20261022/fixture.pt',map_location='cpu',weights_only=True)
    visual=torch.cat([r['visual'] for r in fixture['chunks']])[None].to(runtime.device)
    audio=fixture['audio_info']['embeddings'][None].to(runtime.device)
    time=fixture['timestamps'][None].to(runtime.device)
    # Legacy-equivalence test supplies the captured encoder states explicitly;
    # actual deployed availability still comes from causal PCM/PTS checks.
    available=torch.ones(time.shape,device=runtime.device,dtype=torch.bool)
    stream=AuxiliaryStream(audio,time[:,:,None].expand(audio.shape[:-1]),available,'audio')
    router=SelectedAudioQueryRouter(runtime.audio_router,runtime.vision_router)
    z,p=router(visual,stream,time,available)
    if not torch.equal(z[0].cpu(),torch.cat([r['latents'] for r in fixture['chunks']])):
        raise AssertionError('g_audio=1 did not exactly reproduce native legacy values')
    if not torch.equal(p[0].cpu(),torch.cat([r['attention'] for r in fixture['chunks']])):
        raise AssertionError('g_audio=1 did not exactly reproduce native legacy P')
    original=runtime.generate(fixture['prompt_ids'],z[0])
    if original['generated_ids']!=fixture['generated_ids'].reshape(-1).tolist():
        raise AssertionError('actual g_audio=1 generated IDs differ from legacy capture')
    off,op=router(visual,stream,time,torch.zeros_like(available))
    missing,mp=router(visual,None,time,available)
    vz,vp=runtime.vision_router(visual[0],None,time[0])
    if not torch.equal(off[0],vz) or not torch.equal(missing[0],vz) or not torch.equal(mp[0],vp):
        raise AssertionError('all-off/missing modality fallback changed')
    fallback=runtime.generate(fixture['prompt_ids'],off[0])
    future=AuxiliaryStream(torch.cat((audio,torch.ones((*audio.shape[:2],2,audio.shape[-1]),device=runtime.device)*1000),2),
        torch.cat((stream.timestamps,time[:,:,None].expand(*time.shape,2)+1.),2),available,'audio')
    fz,fp=router(visual,future,time,available)
    if not torch.equal(fz,z) or not torch.equal(fp,p):
        raise AssertionError('future encoded states changed current routing')
    return {'actual_legacy_generated_ids':original['generated_ids'],'vision_fallback_generated_ids':fallback['generated_ids'],
        'g_audio_one_routing_and_generation_exact':True,'all_off_and_missing_routing_exact':True,
        'future_states_removed_before_gate_and_query_attention':True,'native_tokens_per_frame':64,
        'all_baseline_parameters_frozen':all(not p.requires_grad for m in
            (runtime.model,runtime.audio_encoder,runtime.audio_router,runtime.vision_router) for p in m.parameters())}


def integrate(manifest,paired,model_dir,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    seed_all(1234)
    runtime=StrictStreamingRuntime()
    reference={r['question_id']:r for r in read_jsonl(paired)}
    source=json.loads((Path(paired).parent/'paired_config.json').read_text())
    if source['runtime']!=runtime.metadata:raise ValueError('integration encoder/profile changed')
    lock=json.loads((Path(model_dir)/'selection.json').read_text());entry=lock['primary']
    gate,payload_gate=load_gate2(entry['checkpoint'],runtime.device)
    threshold=json.loads(Path(entry['threshold_file']).read_text())
    if threshold['model_sha256']!=sha256(entry['checkpoint']) or threshold['fit_split']!='val':
        raise ValueError('frozen validation threshold mismatch')
    router=SelectedAudioQueryRouter(runtime.audio_router,runtime.vision_router)
    write_json(output/'M2_protocol_integrity.json',verify_protocol(runtime))
    path=output/'gated_results.jsonl'
    previous=read_jsonl(path) if path.exists() else []
    done={r['question_id'] for r in previous}
    if any(r['model_sha256']!=threshold['model_sha256'] for r in previous):raise ValueError('resume model changed')
    for row in [r for r in read_jsonl(manifest) if r['split']=='test']:
        if row['question_id'] in done:continue
        ref=reference[row['question_id']]
        cached=torch.load(ref['feature_cache_path'],map_location='cpu',weights_only=True)
        features=features_from_cache(cached,payload_gate['layout'])
        current=take_features(features,slice(-1,None))
        # Prefix proxy: a candidate exists if any observation in that prefix
        # passed availability. Strict segment deployment uses CURRENT validity.
        current=GateFeatures(current.visual,current.audio,current.context,cached['audio_valid'].any().reshape(1)).to(runtime.device)
        prediction=gate.predict(current)
        accepted=bool(prediction.valid_audio[0] and prediction.utility_score[0]>threshold['threshold'])
        stamps=cached['timestamps'][None].to(runtime.device)
        selection=torch.full(stamps.shape,accepted,device=runtime.device,dtype=torch.bool)
        z,p=router(cached['visual'][None].to(runtime.device),stream_from_payload(cached,runtime.device),stamps,selection)
        generation=runtime.generate(runtime.prompt(row),z[0])
        expected=ref['branches']['audio' if accepted else 'vision']
        if generation['generated_ids']!=expected['generated_ids']:
            raise AssertionError('selected qualified endpoint generation changed')
        scored=score_answer(generation,row['answer'],row['dataset'])
        result={'question_id':row['question_id'],'source_video_id':row['source_video_id'],'task_type':row['task_type'],
            'question_modality':row['question_modality'],'accepted_prefix':accepted,
            'gates':gate.last_output.gates.cpu().tolist(),'availability':cached['audio_valid'].tolist(),
            'selected_windows':(cached['audio_valid']&accepted).tolist(),
            'utility_score':float(prediction.utility_score[0]),'threshold':threshold['threshold'],
            'correct':scored['correct'],'gold_nll':scored['gold_nll'],'prediction':scored['prediction'],
            'generated_ids':generation['generated_ids'],'visual_tokens':generation['visual_tokens'],
            'model_sha256':threshold['model_sha256'],'endpoint_generation_exact':True,
            'one_VLM_generation_for_this_policy':True,'decision_scope':'whole-prefix feasibility proxy at query time',
            'not_a_measured_frame_utility_label':True,'llm_prefill_s':generation['llm_prefill_s']}
        append_jsonl(path,result)
    records=read_jsonl(path)
    write_json(output/'summary.json',{'questions':len(records),'accuracy':sum(r['correct'] for r in records)/len(records),
        'prefix_gate_on_rate':sum(r['accepted_prefix'] for r in records)/len(records),
        'window_gate_on_rate':sum(sum(r['selected_windows']) for r in records)/sum(len(r['selected_windows']) for r in records),
        'endpoint_generations_all_exact':True,'single_VLM_generation_per_policy':True,
        'mode':'prefix feasibility proxy; separate measured-window model required for strict streaming claims'})


def same_checkpoint_null_control(manifest,paired,output):
    """Input ablation: same audio/time query network, no observed audio input.

    This is an additional diagnostic, not a newly trained or silently replaced
    vision fallback. It controls architecture, checkpoint and time context.
    """
    output=Path(output)
    if output.exists():raise FileExistsError('null-input control output is immutable')
    output.mkdir(parents=True)
    seed_all(1234);runtime=StrictStreamingRuntime()
    silent=runtime.audio_encoder.encode_windows(torch.zeros(1,32000),runtime.device).to(runtime.device).float()
    refs={r['question_id']:r for r in read_jsonl(paired)}
    results=[]
    with torch.inference_mode():
        for row in [r for r in read_jsonl(manifest) if r['split']=='test']:
            ref=refs[row['question_id']]
            cache=torch.load(ref['feature_cache_path'],map_location='cpu',weights_only=True)
            visual=cache['visual'].to(runtime.device)
            constant=silent.expand(len(visual),-1,-1)
            z,p=runtime.audio_router(visual,constant,cache['timestamps'].to(runtime.device))
            if not torch.equal(z,p@visual):raise AssertionError('null-input ablation changed native values')
            scored=score_answer(runtime.generate(runtime.prompt(row),z),row['answer'],row['dataset'])
            result={'question_id':row['question_id'],'source_video_id':row['source_video_id'],
                'question_modality':row['question_modality'],'correct':scored['correct'],'gold_nll':scored['gold_nll'],
                'generated_ids':scored['generated_ids'],'visual_tokens':scored['visual_tokens'],
                'actual_audio_correct':ref['branches']['legacy_audio']['correct'],
                'actual_audio_gold_nll':ref['branches']['legacy_audio']['gold_nll'],
                'vision_correct':ref['branches']['vision']['correct'],
                'same_checkpoint_and_time_network':True,'observed_audio_used':False}
            append_jsonl(output/'null_audio_results.jsonl',result);results.append(result)
    write_json(output/'summary.json',{'n':len(results),'null_audio_accuracy':sum(r['correct'] for r in results)/len(results),
        'actual_audio_accuracy':sum(r['actual_audio_correct'] for r in results)/len(results),
        'vision_stage1_accuracy':sum(r['vision_correct'] for r in results)/len(results),
        'actual_audio_only_win':sum(r['actual_audio_correct'] and not r['correct'] for r in results),
        'null_audio_only_win':sum(r['correct'] and not r['actual_audio_correct'] for r in results),
        'same_checkpoint_and_architecture':True,'separate_no_audio_training_performed':False,
        'definition':'precomputed frozen BEATs two-second all-zero waveform states, shared at every timestamp',
        'primary_gate_fallback_unchanged':True,'interpretation':'input-removal diagnostic; not proof of a matched no-audio retraining ablation'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['qualify','integrate','null-control'])
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--paired',type=Path,required=True)
    parser.add_argument('--model-dir',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.stage=='qualify':qualify_endpoints(args.manifest,args.paired,args.output)
    elif args.stage=='integrate':integrate(args.manifest,args.paired,args.model_dir,args.output)
    else:same_checkpoint_null_control(args.manifest,args.paired,args.output)


if __name__=='__main__':main()
