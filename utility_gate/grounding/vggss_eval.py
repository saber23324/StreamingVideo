"""Frozen annotated-frame spatial routing diagnostics, never bbox-guided inference."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path

import numpy as np
import torch

from utility_gate.common import append_jsonl, read_jsonl, seed_all, write_json
from utility_gate.gate2_temporal import Gate2OnlineState
from utility_gate.strict_runtime import StrictStreamingRuntime
from .dataset_audit import ROOT
from .native import encode_times
from .runtime import gates, lock_evaluation
from .spatial_mapping import ImageTransform, LAYOUT, pooled_occupancy, routing_metrics
from .stats import mean_ci


def summarize(rows):
    sources = [r['source_video_id'] for r in rows]
    result = {'n': len(rows), 'sources': len(set(sources)), 'methods': {}}
    if not rows: return result
    for method in rows[0]['methods']:
        result['methods'][method] = {key: mean_ci([r['methods'][method][key] for r in rows], sources)
            for key in ('mass', 'excess_mass', 'pointing')}
    for name, a, b in [('AV_minus_V', 'legacy_audio_64', 'vision_only_64'),
                       ('real_minus_same_ckpt_null', 'legacy_audio_64', 'stage3_null_audio_64'),
                       ('Gate_minus_V', 'gate2_64', 'vision_only_64')]:
        result[name] = mean_ci([r['methods'][a]['mass']-r['methods'][b]['mass'] for r in rows], sources)
        result[name+'_pointing'] = mean_ci([int(r['methods'][a]['pointing'])-int(r['methods'][b]['pointing']) for r in rows], sources)
    result['area_baseline'] = mean_ci([r['area_baseline'] for r in rows], sources)
    return result


def overlay(path, row, frame, maps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    names = ['original', 'vision_only_64', 'legacy_audio_64', 'gate2_64', 'stage3_null_audio_64']
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), constrained_layout=True)
    width, height = frame.shape[1], frame.shape[0]
    ceiling = max(float(m.max()) for m in maps.values())
    for ax, name in zip(axes, names):
        ax.imshow(frame)
        if name != 'original':
            # Display on the valid378x378 field, not stretched across384.
            ax.imshow(maps[name].reshape(14, 14), extent=(0, width*378/384, height*378/384, 0),
                      interpolation='nearest', alpha=.6, cmap='magma', vmin=0, vmax=ceiling)
        for x0, y0, x1, y1 in row['bbox']:
            ax.add_patch(Rectangle((x0*width, y0*height), (x1-x0)*width, (y1-y0)*height,
                                   fill=False, edgecolor='lime', linewidth=2))
        mass = '' if name == 'original' else f"\nROI mass {row['methods'][name]['mass']:.4f}"
        ax.set_title(name+mass, fontsize=10)
        ax.set(xlim=(0, width), ylim=(height, 0)); ax.axis('off')
    fig.suptitle(f"{row['clip_id']} | {row['class']} | t=5, PTS={row['frame_pts_s']:.4f} | valid={row['audio_available']} | area={row['area_baseline']:.4f}")
    fig.savefig(path, dpi=140); plt.close(fig)


def plots(output, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not rows: return
    av = [r['methods']['legacy_audio_64']['mass'] for r in rows]
    vis = [r['methods']['vision_only_64']['mass'] for r in rows]
    null = [r['methods']['stage3_null_audio_64']['mass'] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].scatter(vis, av, s=22, alpha=.7)
    axes[0].plot([0, 1], [0, 1], '--', color='gray')
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel='Vision ROI mass', ylabel='Audio ROI mass', title=f'Valid paired sources, n={len(rows)}')
    axes[1].boxplot([np.asarray(av)-vis, np.asarray(av)-null], tick_labels=['AV minus V', 'Real minus null'], showmeans=True)
    axes[1].axhline(0, color='gray', linestyle='--')
    axes[1].set(ylabel='Paired ROI mass difference', title='Cross-checkpoint / same-checkpoint controls')
    fig.savefig(output/'paired_spatial.png', dpi=170); plt.close(fig)


def run(root=ROOT):
    root = Path(root); output = root/'spatial'; output.mkdir(exist_ok=True)
    lock_evaluation(root, 'vggss')
    seed_all(1234); runtime = StrictStreamingRuntime()
    _, _, gate, threshold = gates(runtime.device)
    silent = runtime.audio_encoder.encode_windows(torch.zeros(1, 32000), runtime.device).to(runtime.device).float()
    (output/'figures').mkdir(exist_ok=True)
    previous = read_jsonl(output/'per_sample.jsonl') if (output/'per_sample.jsonl').exists() else []
    done = {r['clip_id'] for r in previous}
    failures = []
    rows = read_jsonl(root/'datasets/vggss_manifest.jsonl')
    with torch.inference_mode():
        for i, row in enumerate(rows):
            if row['clip_id'] in done: continue
            try:
                if not row['bbox']: raise ValueError('empty_bbox')
                payload, frames = encode_times(runtime, row['media_path'], [2., 4., 5.])
                error = 5-payload['frame_pts_s'][-1]
                if error < -1e-6 or error > row['annotation_tolerance_s']:
                    raise ValueError('annotation_frame_pts_error='+str(error))
                frame = frames[-1][0]
                transform = ImageTransform(frame.shape[1], frame.shape[0])
                occupancy = pooled_occupancy(row['bbox'], transform)
                if occupancy.sum() <= 0: raise ValueError('bbox_has_no_consumed_visual_support')
                _, p_null = runtime.audio_router(payload['visual'], silent.expand(3, -1, -1), payload['timestamps'])
                state = Gate2OnlineState(runtime, gate, threshold)
                p_gate = None
                for j, t in enumerate([2., 4., 5.]):
                    _, p_gate = state.step(payload['visual'][j], payload['causal_audio'][j], t,
                        bool(payload['audio_valid'][j]), t, payload['frame_pts_s'][j])
                matrices = {'vision_only_64': payload['vision_P'][-1], 'legacy_audio_64': payload['audio_P'][-1],
                    'qualified_audio_64': payload['audio_P'][-1] if bool(payload['audio_valid'][-1]) else payload['vision_P'][-1],
                    'gate2_64': p_gate, 'stage3_null_audio_64': p_null[-1],
                    'visual_uniform': torch.full((64, 196), 1/196, device=runtime.device)}
                arrays = {k: v.cpu().numpy() for k, v in matrices.items()}
                result = {k: row[k] for k in ('clip_id', 'source_video_id', 'clip_offset_s', 'class', 'bbox', 'media_sha256')}
                result.update(methods={k: routing_metrics(v, occupancy, row['bbox'], transform) for k, v in arrays.items()},
                    area_baseline=float(occupancy.mean()), bbox_count=len(row['bbox']),
                    frame_pts_s=payload['frame_pts_s'][-1], annotation_time_s=5., pts_error_s=error,
                    audio_available=bool(payload['audio_valid'][-1]), audio_quality=payload['audio_quality'][-1],
                    audio_start_s=3., audio_end_s=5., gate=state.trace[-1],
                    evaluation_mode='annotated_frame_spatial_diagnostic', width=frame.shape[1], height=frame.shape[0],
                    bbox_passed_to_model=False, annotation_time_provenance=row['time_provenance'])
                append_jsonl(output/'per_sample.jsonl', result)
                # Deterministic index choice, never choose figures by favorable mass.
                if i < 12:
                    overlay(output/'figures'/(row['clip_id']+'.png'), result, frame,
                            {k: v.mean(0) for k, v in arrays.items() if k != 'visual_uniform'})
            except Exception as error:
                failure = {'clip_id': row['clip_id'], 'source_video_id': row['source_video_id'], 'reason': str(error)}
                failures.append(failure); append_jsonl(output/'failures.jsonl', failure)
            if (i+1) % 8 == 0: print({'spatial_attempted': i+1, 'failures_this_run': len(failures)}, flush=True)
    results = read_jsonl(output/'per_sample.jsonl') if (output/'per_sample.jsonl').exists() else []
    valid = [r for r in results if r['audio_available']]
    grouped = defaultdict(list)
    for r in valid:
        area = r['area_baseline']
        grouped['area:'+('0-.1' if area <= .1 else '.1-.25' if area <= .25 else '.25-.5' if area <= .5 else '.5-1')].append(r)
        grouped['boxes:'+('1' if r['bbox_count'] == 1 else '2' if r['bbox_count'] == 2 else '3+')].append(r)
        grouped['class:'+r['class']].append(r)
    summary = {'attempted_media': len(rows), 'spatial_valid': len(results), 'valid_audio_primary': summarize(valid),
        'all_quality_legacy_diagnostic': summarize(results), 'quality_counts': dict(Counter(r['audio_quality'] for r in results)),
        'groups_valid_audio': {k: summarize(v) for k, v in grouped.items()}, 'layout': LAYOUT,
        'gate_on_count_valid_audio': sum(r['gate']['accepted'] for r in valid),
        'primary_quality': 'same available-audio paired denominator for all policies',
        'scope': 'positional ROI association; not source segmentation or future QA utility',
        'annotation_frames_never_used_to_select_thresholds': True}
    write_json(output/'summary.json', summary)
    fields = ['clip_id', 'source_video_id', 'audio_quality', 'area_baseline', 'method', 'mass', 'excess_mass', 'pointing', 'lift']
    with (output/'per_sample.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore'); writer.writeheader()
        for row in results:
            for method, value in row['methods'].items(): writer.writerow({**row, 'method': method, **value})
    plots(output, valid)
    print({'spatial_valid': len(results), 'valid_audio': len(valid), 'AV_minus_V': summary['valid_audio_primary'].get('AV_minus_V')})


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, default=ROOT)
    run(p.parse_args().output)


if __name__ == '__main__': main()
