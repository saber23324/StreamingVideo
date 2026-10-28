"""Post-hoc numerical audit of batched versus causal all-off vision controls.

No policy/model selection: the complete previously fixed197-QA cohort is used.
"""
import json
from pathlib import Path

import torch

from utility_gate.common import append_jsonl, read_jsonl, seed_all, sha256, write_json
from utility_gate.gate2_temporal import cached_stream
from utility_gate.runtime import score_answer
from utility_gate.strict_runtime import StrictStreamingRuntime
from .dataset_audit import ROOT, read
from .runtime import gates
from .stats import mean_ci


def run(root=ROOT):
    root = Path(root); rows = read_jsonl(root/'avqa/per_qa.jsonl')
    dataset = {r['question_id']: r for name in ('avqa', 'avqa_supplement')
               for r in read_jsonl(root/'datasets'/(name+'_manifest.jsonl'))}
    config = {'scope': 'post-hoc numerical control, all previously fixed QA; no outcome-based filtering',
        'ids': [r['question_id'] for r in rows], 'source_sha256': sha256(__file__),
        'models': read(root/'pretest_lock_avqa.json')['model_assets']}
    write_json(root/'avqa/numerical_control_lock.json', config)
    seed_all(1234); runtime = StrictStreamingRuntime()
    _, _, gate, threshold = gates(runtime.device)
    results = []
    with torch.inference_mode():
        for i, row in enumerate(rows):
            p = torch.load(row['policy']['cache_path'], map_location=runtime.device, weights_only=True)
            baseline = p['vision_latents']
            state = cached_stream(runtime, gate, threshold, p, force=[False]*len(p['visual']))
            current = state.bank()
            dtype = runtime.model.get_input_embeddings().weight.dtype
            same_input = torch.equal(current.to(dtype), baseline.to(dtype))
            if same_input:
                generation = row['branches']['vision_only_64']; reused = 'identical final-VLM-dtype visual inputs'
            elif not any(row['policy']['stream_selected']):
                other = cached_stream(runtime, gate, threshold, p)
                assert torch.equal(current, other.bank())
                generation = row['branches']['gate2_stream_64']; reused = 'exact same all-off stream bank previously generated'
            else:
                source = dataset[row['question_id']]
                generation = score_answer(runtime.generate(runtime.prompt(source), current), row['gold_answer'], source['dataset'])
                reused = None
            result = {'question_id': row['question_id'], 'source_video_id': row['source_video_id'],
                'max_abs_raw_latent_delta': float((current-baseline).abs().max()),
                'raw_latents_exact': torch.equal(current, baseline), 'final_VLM_dtype_inputs_exact': same_input,
                'causal_vision_correct': generation['correct'], 'batched_vision_correct': row['branches']['vision_only_64']['correct'],
                'generated_ids_exact': generation['generated_ids'] == row['branches']['vision_only_64']['generated_ids'],
                'generation_reuse_reason': reused}
            append_jsonl(root/'avqa/numerical_control.jsonl', result); results.append(result)
            if (i+1) % 32 == 0: print({'checked_alloff': i+1}, flush=True)
    sources = [r['source_video_id'] for r in results]
    summary = {'n': len(results), 'batched_vision_correct': sum(r['batched_vision_correct'] for r in results),
        'causal_vision_correct': sum(r['causal_vision_correct'] for r in results),
        'all_generated_ids_equal': all(r['generated_ids_exact'] for r in results),
        'max_abs_latent_delta': max(r['max_abs_raw_latent_delta'] for r in results),
        'final_VLM_input_exact_count': sum(r['final_VLM_dtype_inputs_exact'] for r in results),
        'new_VLM_generations': sum(r['generation_reuse_reason'] is None for r in results),
        'paired_accuracy_difference': mean_ci([int(r['causal_vision_correct'])-int(r['batched_vision_correct']) for r in results], sources),
        'posthoc_numerical_diagnostic_only': True}
    controls = {}
    for relation in ('ALL', 'View', 'Sound', 'Both'):
        selected = rows if relation == 'ALL' else [r for r in rows if r['question_modality'] == relation]
        ss = [r['source_video_id'] for r in selected]
        a = [r['branches']['legacy_audio_64'] for r in selected]; b = [r['branches']['stage3_null_audio_64'] for r in selected]
        controls[relation] = {'n': len(selected), 'real_only_correct': sum(x['correct'] and not y['correct'] for x, y in zip(a, b)),
            'null_only_correct': sum(y['correct'] and not x['correct'] for x, y in zip(a, b)),
            'paired_real_minus_null_accuracy': mean_ci([int(x['correct'])-int(y['correct']) for x, y in zip(a, b)], ss),
            'paired_real_minus_null_NLL_gain': mean_ci([y['gold_nll']-x['gold_nll'] for x, y in zip(a, b)], ss)}
    summary['same_checkpoint_input_control'] = controls
    write_json(root/'avqa/numerical_control_summary.json', summary)
    print(json.dumps(summary))


if __name__ == '__main__': run()
