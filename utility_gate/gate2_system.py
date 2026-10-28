"""Raw chronological Gate2 verification and single-encoder pipeline timings."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from .common import append_jsonl,fingerprint,read_jsonl,seed_all,sha256,write_json
from .features import GateFeatures,take_features
from .gate2_interfaces import AuxiliaryStream,SelectedAudioQueryRouter,layout_features,load_gate2
from .gate2_runtime import verify_protocol
from .gate2_temporal import Gate2OnlineState,cached_stream
from .runtime import clocked,score_answer,synchronize
from .strict_runtime import StrictStreamingRuntime


def warm_encoders(runtime):
    # Engineering warmup tensors are never dataset rows or QA results.
    runtime.extractor.encode_visual(runtime.model,runtime.processor,np.zeros((1,384,384,3),dtype=np.uint8),1)
    runtime.audio_encoder.encode_windows(torch.zeros(1,32000),runtime.device)
    verify_protocol(runtime)


def raw_replay(manifest,paired,calibration,output,limit=16):
    from relevance_gate.media import frame_at_or_before
    output=Path(output)
    if output.exists():raise FileExistsError('raw stream verification is immutable')
    output.mkdir(parents=True)
    runtime=StrictStreamingRuntime();seed_all(1234);warm_encoders(runtime)
    config=json.loads(Path(calibration).read_text())
    if config['fit_split']!='val' or sha256(config['model_checkpoint'])!=config['model_sha256']:
        raise ValueError('raw replay model/threshold mismatch')
    gate,gate_payload=load_gate2(config['model_checkpoint'],runtime.device)
    if gate_payload['mode']!='measured_segment':raise ValueError('whole-prefix labels cannot claim measured frame utility')
    refs={r['question_id']:r for r in read_jsonl(paired)}
    rows=sorted([r for r in read_jsonl(manifest) if r['split']=='test'],key=lambda r:fingerprint([1234,'raw-replay',r['question_id']]))[:limit]
    results=[]
    for row in rows:
        cache=torch.load(refs[row['question_id']]['feature_cache_path'],map_location='cpu',weights_only=True)
        expected=cached_stream(runtime,gate,config['threshold'],cache)
        live=Gate2OnlineState(runtime,gate,config['threshold'])
        runtime.strict_audio.clear()
        timing=Counter();synchronize();started=time.perf_counter()
        for stamp in np.arange(2.,row['query_time_seconds']+1e-8,2.):
            (frame,pts),elapsed=clocked(lambda:frame_at_or_before(row['video_path'],float(stamp)))
            timing['visual_decode_s']+=elapsed
            visual,elapsed=clocked(lambda:runtime.extractor.encode_visual(runtime.model,runtime.processor,frame[None],1).float()[0])
            timing['visual_encoder_s']+=elapsed
            (windows,valid,reasons,rms),elapsed=clocked(lambda:runtime.causal_windows(row['video_path'],[stamp]))
            timing['audio_decode_s']+=elapsed
            if bool(valid[0]):
                audio,elapsed=clocked(lambda:runtime.audio_encoder.encode_windows(windows,runtime.device).to(runtime.device).float()[0])
                timing['audio_encoder_s']+=elapsed
            else:audio=None
            before=live.bank().clone() if live.latents else None
            _,elapsed=clocked(lambda:live.step(visual,audio,float(stamp),bool(valid[0]),frame_pts=pts))
            timing['gate_and_routing_s']+=elapsed
            if before is not None and not torch.equal(before,live.bank()[:-1]):
                raise AssertionError('future observation changed a historical token bank')
            live.trace[-1].update(quality_reason=reasons[0])
        prompt,tokenize_s=clocked(lambda:runtime.prompt(row))
        generation=runtime.generate(prompt,live.bank())
        processing=time.perf_counter()-started
        scored=score_answer(generation,row['answer'],row['dataset'])
        check=runtime.generate(prompt,expected.bank())
        record={'question_id':row['question_id'],'source_video_id':row['source_video_id'],
            'correct':scored['correct'],'generated_ids':generation['generated_ids'],
            'raw_cached_generated_ids_equal':generation['generated_ids']==check['generated_ids'],
            'raw_cached_selections_equal':[r['accepted'] for r in live.trace]==[r['accepted'] for r in expected.trace],
            'trace':live.trace,'history_never_revised':True,'no_future_question_input':True,
            'question_tokenization_s':tokenize_s,'llm_prefill_s':generation['llm_prefill_s'],
            'question_ttft_s':tokenize_s+generation['llm_ttft_s'],
            'processing_s_excluding_arrival_wait':processing,'timing':dict(timing),
            'single_VLM_generation_in_deployed_policy':True,
            'cached_verification_is_separate_and_excluded_from_policy_time':True}
        append_jsonl(output/'raw_replay.jsonl',record);results.append(record)
        print(json.dumps({'raw_verified':row['question_id'],'tokens_equal':record['raw_cached_generated_ids_equal']}),flush=True)
    write_json(output/'summary.json',{'n':len(results),'accuracy':float(np.mean([r['correct'] for r in results])),
        'raw_cached_generations_all_equal':all(r['raw_cached_generated_ids_equal'] for r in results),
        'raw_cached_selections_all_equal':all(r['raw_cached_selections_equal'] for r in results),
        'historical_tokens_never_revised':True,'encoder_costs_included':True,
        'scope':'raw per-arrival media and one QA call; simulated arrival clock, no real-time sleeping'})


def prefix_pipeline(runtime,row,gate,threshold,energy_threshold,method):
    from relevance_gate.media import frame_at_or_before
    runtime.strict_audio.clear();torch.cuda.reset_peak_memory_stats()
    synchronize();started=time.perf_counter()
    stamps=np.arange(2.,row['query_time_seconds']+1e-8,2.,dtype=np.float32)
    def decode():return np.stack([frame_at_or_before(row['video_path'],float(t))[0] for t in stamps])
    frames,decode_s=clocked(decode)
    visual,vision_s=clocked(lambda:runtime.extractor.encode_visual(runtime.model,runtime.processor,frames,1).float())
    timing={'visual_decode_s':decode_s,'visual_encoder_s':vision_s,'audio_decode_s':0.,'audio_encoder_s':0.,'gate_s':0.,'routing_s':0.}
    timestamps=torch.tensor(stamps,device=runtime.device)[None]
    audio=None;valid=torch.zeros(len(stamps),dtype=torch.bool)
    accepted=method in ('legacy_audio','safe_audio')
    if method in ('legacy_audio','safe_audio','learned','energy'):
        (windows,valid,reasons,rms),timing['audio_decode_s']=clocked(lambda:runtime.causal_windows(row['video_path'],stamps))
        if method=='energy':accepted=bool(valid.any()) and float(torch.log1p(rms.mean()))>energy_threshold
        if method!='energy' or accepted:
            def encode():
                if method=='legacy_audio':return runtime.audio_encoder.encode_windows(windows,runtime.device).to(runtime.device).float()
                if not bool(valid.any()):return torch.zeros((len(stamps),1,768),device=runtime.device)
                states=runtime.audio_encoder.encode_windows(windows[valid],runtime.device).to(runtime.device).float()
                result=torch.zeros((len(stamps),states.shape[1],states.shape[2]),device=runtime.device)
                result[valid.to(runtime.device)]=states
                return result
            audio,timing['audio_encoder_s']=clocked(encode)
    if audio is not None:
        stream=AuxiliaryStream(audio[None],timestamps[:,:,None].expand(1,len(stamps),audio.shape[1]),valid[None].to(runtime.device),'audio')
    else:stream=None
    if method=='learned':
        def decide():
            features=layout_features(visual[None],timestamps,stream,gate.layout)
            current=take_features(features,slice(-1,None))
            current=GateFeatures(current.visual,current.audio,current.context,valid.any().reshape(1).to(runtime.device))
            prediction=gate.predict(current)
            return bool(prediction.valid_audio[0] and prediction.utility_score[0]>threshold)
        accepted,timing['gate_s']=clocked(decide)
    if method=='full_vision':latents=runtime.full_vision(visual)
    elif method=='legacy_audio':
        (latents,_),timing['routing_s']=clocked(lambda:runtime.audio_router(visual,audio,timestamps[0]))
    elif not accepted:
        (latents,_),timing['routing_s']=clocked(lambda:runtime.vision_router(visual,None,timestamps[0]))
    else:
        router=SelectedAudioQueryRouter(runtime.audio_router,runtime.vision_router)
        (bank,_),timing['routing_s']=clocked(lambda:router(visual[None],stream,timestamps,torch.ones_like(timestamps,dtype=torch.bool)))
        latents=bank[0]
    prompt=runtime.prompt(row)
    before_llm=time.perf_counter()-started
    generated=runtime.generate(prompt,latents)
    synchronize()
    return {'question_id':row['question_id'],'method':method,'accepted_prefix':accepted,**timing,
        'llm_prefill_s':generated['llm_prefill_s'],'llm_total_s':generated['llm_total_s'],
        'pipeline_ttft_s':before_llm+generated['llm_ttft_s'],'pipeline_total_s':time.perf_counter()-started,
        'visual_tokens':generated['visual_tokens'],'generated_ids':generated['generated_ids'],
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'one_audio_frontend_shared_by_gate_and_router':True,'encoder_cost_included_on_rejected_audio':True}


def benchmark(manifest,model_dir,output,limit=8):
    output=Path(output)
    if output.exists():raise FileExistsError('pipeline timing output is immutable')
    output.mkdir(parents=True)
    runtime=StrictStreamingRuntime();seed_all(1234);warm_encoders(runtime)
    selection=json.loads((Path(model_dir)/'selection.json').read_text());primary=selection['primary']
    gate,payload=load_gate2(primary['checkpoint'],runtime.device)
    cutoff=json.loads(Path(primary['threshold_file']).read_text())
    if cutoff['fit_split']!='val' or cutoff['model_sha256']!=sha256(primary['checkpoint']):raise ValueError('timing used an unfrozen gate')
    methods=('full_vision','always_off','legacy_audio','safe_audio','energy','learned')
    rows=sorted([r for r in read_jsonl(manifest) if r['split']=='test'],key=lambda r:fingerprint([1234,'gate2-latency',r['question_id']]))[:limit]
    results=[]
    with torch.inference_mode():
        for row in rows:
            order=sorted(methods,key=lambda name:fingerprint([row['question_id'],name]))
            for method in order:
                result=prefix_pipeline(runtime,row,gate,cutoff['threshold'],selection['energy_threshold']['threshold'],method)
                append_jsonl(output/'latency_breakdown.jsonl',result);results.append(result)
                print(json.dumps({'timed':row['question_id'],'method':method,'total_s':result['pipeline_total_s']}),flush=True)
    with (output/'latency_breakdown.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=[k for k in results[0] if k!='generated_ids'],extrasaction='ignore')
        writer.writeheader();writer.writerows(results)
    metrics={method:{key:float(np.median([r[key] for r in results if r['method']==method])) for key in
        ('visual_decode_s','visual_encoder_s','audio_decode_s','audio_encoder_s','gate_s','routing_s',
         'llm_prefill_s','pipeline_ttft_s','pipeline_total_s','peak_allocated_bytes')} for method in methods}
    write_json(output/'summary.json',{'n':len(rows),'methods':metrics,'warm_model':True,
        'counts_audio_encoder_even_if_gate_rejects':True,'tensor_feature_cache_used':False,
        'excludes':['model loading','network downloading','cold OS file cache','arrival waiting'],
        'resident_modules_shared_across_methods':True,'hardware':'shared H800 node; bounded observational measurements'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['raw','benchmark'])
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--paired',type=Path)
    parser.add_argument('--model-dir',type=Path)
    parser.add_argument('--calibration',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.stage=='raw':raw_replay(args.manifest,args.paired,args.calibration,args.output)
    else:benchmark(args.manifest,args.model_dir,args.output)


if __name__=='__main__':main()
