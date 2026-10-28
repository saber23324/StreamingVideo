"""Explicit timestamp extraction, reusing the protected strict audio frontend."""
import numpy as np
import torch

from relevance_gate.media import frame_at_or_before


@torch.inference_mode()
def encode_times(runtime, path, timestamps):
    times = list(map(float, timestamps))
    if not times or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('nonempty increasing observed timestamps required')
    frames = [frame_at_or_before(path, t) for t in times]
    if any(pts > t+1e-6 for (_, pts), t in zip(frames, times)):
        raise ValueError('future visual frame')
    visual = runtime.extractor.encode_visual(runtime.model, runtime.processor, np.stack([f[0] for f in frames]), 1).float()
    windows, valid, quality, rms = runtime.causal_windows(path, times)
    audio = runtime.audio_encoder.encode_windows(windows, runtime.device).to(runtime.device).float()
    stamps = torch.tensor(times, device=runtime.device)
    av, p_av = runtime.audio_router(visual, audio, stamps)
    vis, p_vis = runtime.vision_router(visual, None, stamps)
    if not torch.equal(av, p_av@visual) or not torch.equal(vis, p_vis@visual):
        raise AssertionError('native-value resampling changed')
    return {'visual': visual, 'causal_audio': audio, 'audio_valid': valid,
            'audio_quality': quality, 'audio_rms': rms, 'timestamps': stamps,
            'frame_pts_s': [f[1] for f in frames], 'audio_latents': av,
            'vision_latents': vis, 'audio_P': p_av, 'vision_P': p_vis}, frames
