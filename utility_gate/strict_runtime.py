"""Separate strict-arrival adapter, preserving the completed prefix runtime."""
from __future__ import annotations

import time

import numpy as np
import torch

from .common import sha256
from .features import feature_payload, prefix_features
from .runtime import FrozenRuntime, clocked


class StrictStreamingRuntime(FrozenRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata.update(strict_streaming_source_sha256=sha256(__file__),
            arrival_grid='2,4,6,... seconds, independent of future query; no frame cap',
            initial_tick_s=2.,vision_batch_size=1,
            routing_audio='truncate native audio before resampling at each decision timestamp')

    @torch.inference_mode()
    def encode(self,row,mode='streaming',max_frames=0):
        from relevance_gate.media import frame_at_or_before
        if mode!='streaming':
            raise ValueError('strict adapter is only for chronological window experiments')
        interval=float(row.get('frame_interval_s',2.))
        timestamps=np.arange(interval,float(row['query_time_seconds'])+1e-8,interval,dtype=np.float32)
        if not len(timestamps):
            raise ValueError('query precedes first scheduled observation')
        torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        def decode():
            frames=[frame_at_or_before(row['video_path'],float(t)) for t in timestamps]
            return np.stack([f[0] for f in frames]),[f[1] for f in frames]
        (frames,pts),decode_s=clocked(decode)
        visual,visual_s=clocked(lambda:self.extractor.encode_visual(self.model,self.processor,frames,1).float())
        (windows,valid,reasons,rms),audio_decode_s=clocked(lambda:self.causal_windows(row['video_path'],timestamps))
        audio,audio_s=clocked(lambda:self.audio_encoder.encode_windows(windows,self.device).to(self.device).float())
        times=torch.as_tensor(timestamps,device=self.device)
        (av,av_p),av_s=clocked(lambda:self.audio_router(visual,audio,times))
        (vision,vision_p),vision_s=clocked(lambda:self.vision_router(visual,None,times))
        if visual.shape[1]!=196 or av.shape[1]!=64 or vision.shape!=av.shape:
            raise AssertionError('strict branch budget mismatch')
        if not torch.equal(av,av_p@visual) or not torch.equal(vision,vision_p@visual):
            raise AssertionError('native visual values changed')
        features=prefix_features(visual,audio,valid,times)
        return {'visual':visual.cpu(),'causal_audio':audio.cpu(),'router_audio':audio.cpu(),
            'audio_latents':av.cpu(),'vision_latents':vision.cpu(),'timestamps':times.cpu(),
            'frame_pts_s':pts,'audio_valid':valid,'audio_quality':reasons,'audio_rms':rms,
            'audio_windows':windows,'features':feature_payload(features),'mode':'streaming',
            'timing':{'video_decode_s':decode_s,'visual_encoder_s':visual_s,
                'causal_audio_decode_s':audio_decode_s,'audio_encoder_s':audio_s,
                'legacy_audio_decode_and_encoder_s':0.,'audio_routing_s':av_s,'vision_routing_s':vision_s,
                'two_branch_feature_total_s':time.perf_counter()-started,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes':torch.cuda.max_memory_reserved()}}
