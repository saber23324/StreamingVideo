"""Shared read-only frozen runtimes and pre-evaluation identity locks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from relevance_gate.media import frame_at_or_before
from utility_gate.common import read_jsonl, seed_all, sha256, write_json
from utility_gate.gate2_interfaces import load_gate2
from utility_gate.gate2_runtime import verify_protocol
from utility_gate.integration import verify_disabled
from utility_gate.strict_runtime import StrictStreamingRuntime
from .dataset_audit import DATA_ROOT, OLD, ROOT, protected_assets, read
from .spatial_mapping import LAYOUT


def gates(device):
    prefix = read(OLD/'qualified_model/selection.json')['primary']
    temporal = read(OLD/'temporal_calibration/streaming_threshold.json')
    prefix_model, _ = load_gate2(prefix['checkpoint'], device)
    temporal_model, _ = load_gate2(temporal['model_checkpoint'], device)
    return prefix_model, read(prefix['threshold_file'])['threshold'], temporal_model, temporal['threshold']


def audit_runtime(output=ROOT):
    output = Path(output)
    seed_all(1234)
    runtime = StrictStreamingRuntime()
    m0 = read(OLD/'m0/integrity.json')
    for name in ('audio_router_sha256', 'vision_router_sha256'):
        assert runtime.metadata[name] == m0['runtime_metadata'][name]
    for name in ('vision_state_sha256', 'projector_sha256', 'beats_sha256', 'model_commit'):
        assert runtime.metadata['encoder'][name] == m0['runtime_metadata']['encoder'][name]
    disabled, protocol = verify_disabled(runtime), verify_protocol(runtime)
    previous = read_jsonl(DATA_ROOT/'utility_gate/gate2_20261027/qa_manifest.jsonl')
    # A deterministic pre-existing test fixture, selected without its outcome.
    row = next(r for r in previous if r['split'] == 'test')
    references = {r['question_id']: r for r in read_jsonl(OLD/'qualified_paired/paired_qa.jsonl')}
    ref = references[row['question_id']]
    cached = torch.load(ref['feature_cache_path'], map_location='cpu', weights_only=True)
    actual = runtime.encode(row, mode='streaming')
    comparisons = {}
    for key in ('visual', 'causal_audio', 'audio_latents', 'vision_latents'):
        comparisons[key] = {'exact': torch.equal(cached[key], actual[key]),
                            'max_abs': float((cached[key]-actual[key]).abs().max())}
        if not torch.allclose(cached[key], actual[key], atol=1e-4, rtol=1e-5):
            raise AssertionError('fresh extraction differs from protected cache: '+key)
    generated = {}
    for key, branch in [('audio_latents', 'legacy_audio'), ('vision_latents', 'vision')]:
        result = runtime.generate(runtime.prompt(row), actual[key].to(runtime.device))
        assert result['generated_ids'] == ref['branches'][branch]['generated_ids']
        generated[branch] = result['generated_ids']
    # Verify the actual model's embedding convolution, flatten and pool order.
    tower = runtime.model.get_vision_tower()
    conv = tower.vision_tower.vision_model.embeddings.patch_embedding
    assert conv.kernel_size == (14, 14) and conv.stride == (14, 14) and conv.padding in ('valid', (0, 0))
    from streamingtom.tasks.vision import VisionTask
    coordinate = torch.arange(729, dtype=torch.float32, device=runtime.device).reshape(1, 729, 1)
    pooled = VisionTask(None, None)._apply_2d_pool(coordinate)
    expected = torch.nn.functional.interpolate(coordinate.reshape(1, 27, 27, 1).permute(0, 3, 1, 2),
        size=(14, 14), mode='bilinear', align_corners=False).permute(0, 2, 3, 1).reshape(1, 196, 1)
    assert torch.equal(pooled, expected)
    # A non-square RGB coordinate image verifies direct resize without a crop.
    ramp = np.zeros((180, 320, 3), np.uint8)
    ramp[:, :, 0] = np.arange(320)[None, :]*255/319
    ramp[:, :, 1] = np.arange(180)[:, None]*255/179
    pixels = runtime.processor.preprocess(ramp[None], return_tensors='pt')['pixel_values']
    assert tuple(pixels.shape) == (1, 3, 384, 384)
    restored = pixels[0]*.5+.5
    assert restored[0, :, 0].mean() < .01 and restored[0, :, -1].mean() > .99
    assert restored[1, 0].mean() < .01 and restored[1, -1].mean() > .99
    result = {'disabled': disabled, 'protocol': protocol, 'fresh_old_QA_id': row['question_id'],
        'fresh_extraction_comparison': comparisons, 'fresh_generated_ids': generated,
        'runtime': runtime.metadata, 'layout': LAYOUT, 'actual_tower_conv_checked': True,
        'actual_pool_coordinate_ramp_exact': True, 'actual_non_square_processor_no_crop_checked': True,
        'frozen_all_parameters': all(not p.requires_grad for m in (runtime.model, runtime.audio_encoder,
             runtime.audio_router, runtime.vision_router) for p in m.parameters()),
        'causal_audio_decode': 'truncate native waveform before resampling at each end time'}
    write_json(output/'M0_runtime_integrity.json', result)
    return result


def lock_evaluation(root=ROOT, dataset=None):
    root = Path(root)
    stages = {'vggss': 'vggss_eval.py', 'llp': 'llp_gate_eval.py', 'avqa': 'avqa_group_eval.py'}
    names = ['runtime.py', 'native.py', 'spatial_mapping.py', 'llp_parser.py', 'stats.py']
    paths = [Path(__file__).with_name(n) for n in names]
    paths += [Path(__file__).with_name(stages[dataset]), root/'datasets'/(dataset+'_manifest.jsonl')] if dataset else list((root/'datasets').glob('*_manifest.jsonl'))
    record = {'model_assets': {k: {'path': p, 'sha256': sha256(p)} for k, p in protected_assets().items()},
              'evaluation_sources_and_manifests_sha256': {str(p): sha256(p) for p in paths},
              'all_weights_and_thresholds_preexisting': True, 'no_fit_on_any_grounding_label': True,
              'spatial_temporal_gate': 'existing measured-window model, existing streaming_threshold',
              'AVQA_gate_prefix': 'existing primary prefix model, existing threshold; feasibility proxy',
              'AVQA_gate_stream': 'existing measured-window model, immutable per-arrival decisions'}
    path = root/(('pretest_lock_'+dataset) if dataset else 'pretest_lock')
    path = path.with_suffix('.json')
    if path.exists() and read(path) != record:
        raise ValueError('formal grounding evaluation already locked')
    write_json(path, record)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['audit', 'lock'])
    p.add_argument('--output', type=Path, default=ROOT)
    p.add_argument('--dataset', choices=['vggss', 'llp', 'avqa'])
    args = p.parse_args()
    if args.stage == 'audit':
        result = audit_runtime(args.output)
        print(json.dumps({k: v for k, v in result.items() if k != 'runtime'}))
    else: lock_evaluation(args.output, args.dataset)


if __name__ == '__main__': main()
