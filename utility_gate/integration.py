"""M3 actual single-generation integration plus bounded uncached pipeline timing."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from .common import (REPO, append_jsonl, fingerprint, read_jsonl, seed_all, sha256,
                     write_json, write_jsonl)
from .features import load_feature_payload, prefix_features, take_features
from .models import load_gate
from .policy import UtilityRoutingPolicy
from .runtime import FrozenRuntime, clocked, score_answer, synchronize


def policy_from_selection(runtime, selection_dir, expected_mode='prefix_proxy'):
    selection_dir=Path(selection_dir)
    selection=json.loads((selection_dir/'selection.json').read_text())
    primary=selection['primary']
    gate,payload=load_gate(primary['checkpoint'],runtime.device)
    threshold=json.loads(Path(primary['threshold_path']).read_text())
    if threshold['fit_split'] != 'val' or threshold['model_sha256'] != sha256(primary['checkpoint']):
        raise ValueError('frozen validation threshold/checkpoint mismatch')
    if payload['mode'] != expected_mode:
        raise ValueError('prefix proxy and measured segment gates are not interchangeable')
    return UtilityRoutingPolicy(runtime.audio_router,runtime.vision_router,gate,threshold['threshold'],True), selection


def verify_disabled(runtime):
    fixture=torch.load(REPO/'artifacts/relevance_gate/m0-gate-disabled-final-20261022/fixture.pt',map_location='cpu',weights_only=True)
    v=torch.cat([c['visual'] for c in fixture['chunks']]).to(runtime.device)
    a=fixture['audio_info']['embeddings'].to(runtime.device)
    t=fixture['timestamps'].to(runtime.device)
    disabled=UtilityRoutingPolicy(runtime.audio_router)
    with torch.inference_mode():
        z,p=disabled(v,a,t)
    if not torch.equal(z.cpu(),torch.cat([c['latents'] for c in fixture['chunks']])) or not torch.equal(p.cpu(),torch.cat([c['attention'] for c in fixture['chunks']])):
        raise AssertionError('disabled Utility Gate changed actual routing')
    generation=runtime.generate(fixture['prompt_ids'],z)
    if generation['generated_ids'] != fixture['generated_ids'].reshape(-1).tolist():
        raise AssertionError('disabled Utility Gate changed actual generated tokens')
    return {'disabled_routing_exact':True,'disabled_generated_ids_exact':True,'generated_ids':generation['generated_ids']}


def cached_integration(manifest,paired,selection_dir,output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    all_rows=read_jsonl(manifest)
    records={r['question_id']:r for r in read_jsonl(paired)}
    rows=[r for r in all_rows if r['split']=='test']
    seed_all(1234)
    runtime=FrozenRuntime()
    config=json.loads((Path(paired).parent/'paired_config.json').read_text())
    if config['runtime'] != runtime.metadata:
        raise ValueError('baseline model, encoder, runtime or scorer changed after paired inference')
    policy,selection=policy_from_selection(runtime,selection_dir)
    write_json(output/'regression.json',verify_disabled(runtime))
    path=output/'gated_results.jsonl'
    previous=read_jsonl(path) if path.is_file() else []
    done={r['question_id'] for r in previous}
    if len(done)!=len(previous):
        raise ValueError('duplicate integration resume IDs')
    checkpoint_sha=sha256(selection['primary']['checkpoint'])
    if any(r['gate_sha256']!=checkpoint_sha for r in previous):
        raise ValueError('resume uses a different frozen gate')
    for index,row in enumerate(rows,1):
        if row['question_id'] in done:
            continue
        reference=records[row['question_id']]
        payload=torch.load(reference['feature_cache_path'],map_location='cpu',weights_only=True)
        features=load_feature_payload(payload['features']).to(runtime.device)
        current=take_features(features,slice(-1,None))
        (latents,attention),routing_s=clocked(lambda:policy(payload['visual'].to(runtime.device),
            payload['router_audio'].to(runtime.device),payload['timestamps'].to(runtime.device),current))
        accepted=bool(policy.last_decision.accepted[0])
        expected=payload['audio_latents'] if accepted else payload['vision_latents']
        if not torch.equal(latents.cpu(),expected):
            raise AssertionError('utility routing differs from its whole original endpoint')
        generation=runtime.generate(runtime.prompt(row),latents)
        branch=reference['branches']['audio' if accepted else 'vision']
        if generation['generated_ids']!=branch['generated_ids']:
            raise AssertionError('single selected-branch generation differs from paired endpoint')
        scored=score_answer(generation,row['answer'],row['dataset'])
        result={'question_id':row['question_id'],'source_video_id':row['source_video_id'],'dataset':row['dataset'],
            'task_type':row['task_type'],'accepted':accepted,'utility_score':float(policy.last_decision.utility_score[0]),
            'threshold':policy.threshold,'gate_sha256':checkpoint_sha,'visual_tokens':generation['visual_tokens'],
            'generated_ids':generation['generated_ids'],'prediction':scored['prediction'],'correct':scored['correct'],
            'gold_nll':scored['gold_nll'],'single_vlm_generation':True,'endpoint_generation_exact':True,
            'timing_scope':'cached encoders; actual routing and one VLM generation, NOT end-to-end pipeline',
            'routing_and_gate_s':routing_s,'llm_prefill_s':generation['llm_prefill_s'],
            'llm_ttft_s':generation['llm_ttft_s'],'llm_total_s':generation['llm_total_s']}
        append_jsonl(path,result)
        print(json.dumps({'integrated':index,'total':len(rows),'id':row['question_id'],'accepted':accepted,'correct':scored['correct']}),flush=True)
    results=read_jsonl(path)
    if len(results)!=len(rows):
        raise ValueError('integration incomplete')
    with (output/'gated_results.csv').open('w') as stream:
        columns=['question_id','source_video_id','dataset','task_type','accepted','utility_score','threshold','visual_tokens',
                 'prediction','correct','gold_nll','single_vlm_generation','endpoint_generation_exact']
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(results)
    write_json(output/'summary.json',{'questions':len(results),'accuracy':sum(r['correct'] for r in results)/len(results),
        'gate_on_rate':sum(r['accepted'] for r in results)/len(results),'endpoint_generations_all_exact':True,
        'single_vlm_generation_per_decision':True,'gate_checkpoint_sha256':checkpoint_sha,
        'mode':'prefix_proxy','all_parameters_frozen':all(not p.requires_grad for module in
            (runtime.model,runtime.audio_encoder,runtime.audio_router,runtime.vision_router,policy.gate) for p in module.parameters())})


def pipeline_one(runtime,row,policy,method,energy_threshold):
    from streamingtom.train_adbt_videomme import sample_video_for_rows
    runtime.waveforms.clear();runtime.strict_audio.clear()
    torch.cuda.reset_peak_memory_stats()
    synchronize();started=time.perf_counter()
    timing={k:0. for k in ('audio_decode_s','audio_encoder_s','gate_s','routing_s')}
    (frames,timestamps,_),timing['video_decode_s']=clocked(lambda:sample_video_for_rows(row['video_path'],[row],fps='auto',max_frames=8))
    visual,timing['visual_encoder_s']=clocked(lambda:runtime.extractor.encode_visual(runtime.model,runtime.processor,frames,4).float())
    t=torch.as_tensor(timestamps,device=runtime.device)
    accepted=method=='always_audio'
    audio=None
    if method in ('utility_gate','energy_gate'):
        (windows,valid,_,rms),timing['audio_decode_s']=clocked(lambda:runtime.causal_windows(row['video_path'],timestamps))
        if method=='utility_gate':
            causal,timing['audio_encoder_s']=clocked(lambda:runtime.audio_encoder.encode_windows(windows,runtime.device).to(runtime.device).float())
            def decide():
                features=prefix_features(visual,causal,valid,t)
                return policy.choose(take_features(features,slice(-1,None)))
            decision,timing['gate_s']=clocked(decide)
            accepted=bool(decision.accepted[0])
        else:
            accepted=bool(valid.any()) and float(torch.log1p(rms.mean()))>energy_threshold
    if accepted:
        waveform,decode_s=clocked(lambda:runtime.audio_encoder.decode_audio(row['video_path']))
        windows=runtime.audio_encoder._causal_windows(waveform,torch.as_tensor(timestamps))
        audio,encoder_s=clocked(lambda:runtime.audio_encoder.encode_windows(windows,runtime.device).to(runtime.device).float())
        timing['audio_decode_s']+=decode_s;timing['audio_encoder_s']+=encoder_s
    if method=='full_vision':
        latents=runtime.full_vision(visual)
    elif accepted:
        (latents,_),timing['routing_s']=clocked(lambda:runtime.audio_router(visual,audio,t))
    else:
        (latents,_),timing['routing_s']=clocked(lambda:runtime.vision_router(visual,None,t))
    # Tokenization is part of arrival-to-first-token time as well as total time.
    prompt=runtime.prompt(row)
    before_llm=time.perf_counter()-started
    generation=runtime.generate(prompt,latents)
    synchronize();total=time.perf_counter()-started
    return {'question_id':row['question_id'],'method':method,'accepted':accepted,
        **timing,'llm_prefill_s':generation['llm_prefill_s'],'llm_decode_and_prefill_s':generation['llm_total_s'],
        'pipeline_ttft_s':before_llm+generation['llm_ttft_s'],'pipeline_total_s':total,
        'visual_tokens':generation['visual_tokens'],'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'generated_ids':generation['generated_ids'],
        'no_feature_cache':True,'audio_cost_included_when_rejected':True}


def benchmark(manifest,selection_dir,output,limit=8):
    output=Path(output)
    if output.exists():
        raise FileExistsError('use fresh pipeline timing output')
    output.mkdir(parents=True)
    rows=sorted([r for r in read_jsonl(manifest) if r['split']=='test'],key=lambda r:fingerprint(['latency',1234,r['question_id']]))[:limit]
    seed_all(1234)
    runtime=FrozenRuntime()
    policy,selection=policy_from_selection(runtime,selection_dir)
    verify_disabled(runtime)  # Model/kernel warmup, excluded from measured intervals.
    results=[]
    methods=['full_vision','always_vision','always_audio','utility_gate','energy_gate']
    with torch.inference_mode():
        for row in rows:
            order=sorted(methods,key=lambda name:fingerprint([row['question_id'],name]))
            for method in order:
                result=pipeline_one(runtime,row,policy,method,selection['energy_threshold']['threshold'])
                append_jsonl(output/'latency_breakdown.jsonl',result)
                results.append(result)
                print(json.dumps({'timed':row['question_id'],'method':method,'total_s':result['pipeline_total_s']}),flush=True)
    fields=[key for key in results[0] if key!='generated_ids']
    with (output/'latency_breakdown.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(results)
    metrics={}
    for method in methods:
        selected=[r for r in results if r['method']==method]
        metrics[method]={key:float(np.median([r[key] for r in selected])) for key in
            ('video_decode_s','visual_encoder_s','audio_decode_s','audio_encoder_s','gate_s','routing_s',
             'llm_prefill_s','pipeline_ttft_s','pipeline_total_s','peak_allocated_bytes')}
    write_json(output/'summary.json',{'questions':len(rows),'methods':metrics,
        'scope':'warm model, raw media decoding plus encoders/routing/one greedy VLM generation; no tensor feature cache',
        'exclusions':['model/checkpoint loading','network acquisition','cold OS filesystem cache'],
        'environment_note':'shared H800 node with unrelated jobs; bounded observational timing, not a dedicated-node speed claim',
        'all_audio_encoding_included':True,'prefix_proxy':True,'warmup':'legacy exact fixture before measurements'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--paired',type=Path)
    parser.add_argument('--selection-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--benchmark',action='store_true')
    args=parser.parse_args()
    if args.benchmark:
        benchmark(args.manifest,args.selection_dir,args.output)
    else:
        if args.paired is None:
            parser.error('--paired required for cached integration')
        cached_integration(args.manifest,args.paired,args.selection_dir,args.output)


if __name__=='__main__':
    main()
