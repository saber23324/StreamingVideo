"""PTS-aware media reads and strict causal quality checks for new gate work.

The legacy audio frontend remains unchanged. Only these independent experiments
reject missing, expired, silent or malformed windows explicitly.
"""

from dataclasses import dataclass
import math

import numpy as np
import torch


@dataclass
class AudioWindow:
    samples: torch.Tensor
    sample_mask: torch.Tensor
    valid: bool
    reason: str
    start_s: float
    end_s: float
    rms: float = 0.


def causal_window(waveform, timestamp_s, sample_rate=16000, window_seconds=2.,
                  origin_s=0., start_s=None, min_rms=1e-5):
    if sample_rate <= 0 or not math.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("positive sample rate and window duration required")
    length = int(round(window_seconds * sample_rate))
    if length == 0:
        raise ValueError("window has no samples")
    zeros = torch.zeros(length)
    mask = torch.zeros(length, dtype=torch.bool)
    if not math.isfinite(float(timestamp_s)) or not math.isfinite(float(origin_s)):
        return AudioWindow(zeros, mask, False, "invalid_timestamp", 0., 0.)
    end_s = float(timestamp_s)
    begin = max(0., end_s - window_seconds) if start_s is None else float(start_s)
    if not math.isfinite(begin) or begin < 0 or end_s < begin or end_s - begin > window_seconds + 1e-8:
        return AudioWindow(zeros, mask, False, "invalid_window", begin, end_s)
    if waveform is None or waveform.numel() == 0:
        return AudioWindow(zeros, mask, False, "missing_audio", begin, end_s)
    if waveform.ndim != 1:
        return AudioWindow(zeros, mask, False, "malformed_audio", begin, end_s)
    duration_end = origin_s + waveform.numel() / sample_rate
    if end_s > duration_end + 1. / sample_rate or end_s <= origin_s or end_s <= begin:
        return AudioWindow(zeros, mask, False, "out_of_window", begin, end_s)
    # Integer sample k corresponds to [origin+k/sr, origin+(k+1)/sr).
    # ceil at the start and floor at the end exclude any future sample interval.
    first = max(0, math.ceil((begin - origin_s) * sample_rate - 1e-8))
    last = min(waveform.numel(), math.floor((end_s - origin_s) * sample_rate + 1e-8))
    clip = waveform[first:last].detach().float()
    if not clip.numel():
        return AudioWindow(zeros, mask, False, "empty_window", begin, end_s)
    if clip.numel() > length or not torch.isfinite(clip).all():
        return AudioWindow(zeros, mask, False, "nonfinite_audio", begin, end_s)
    zeros[-clip.numel():] = clip
    mask[-clip.numel():] = True
    rms = float(clip.square().mean().sqrt())
    valid = rms > min_rms
    return AudioWindow(zeros, mask, valid, "ok" if valid else "silence", begin, end_s, rms)


def decode_audio(path, sample_rate=16000, end_s=None):
    """Return mono PCM, origin relative to media time, and PTS diagnostics.

    Missing tracks return None. PTS gaps are zero-filled, preserving elapsed
    time rather than silently concatenating discontinuous audio packets.
    """
    import av
    if end_s is not None and (not math.isfinite(end_s) or end_s < 0):
        raise ValueError("causal audio end must be finite and nonnegative")
    if path is None:
        return None, 0., {"available": False}
    with av.open(str(path)) as container:
        streams = [s for s in container.streams if s.type == "audio"]
        if not streams:
            return None, 0., {"available": False}
        stream = streams[0]
        media_origin = (container.start_time or 0) / av.time_base
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate)
        chunks = []
        for frame in container.decode(stream):
            if end_s is not None:
                if frame.pts is None:
                    return None, 0., {"available": False, "reason": "missing_audio_pts"}
                begin = float(frame.pts * frame.time_base) - media_origin
                if begin >= end_s:
                    break
                allowed = min(frame.samples, max(0, math.floor((end_s - begin) * frame.sample_rate + 1e-8)))
                if allowed <= 0:
                    break
                if allowed < frame.samples:
                    # Truncate BEFORE resampling. A whole-track symmetric
                    # resampling filter could otherwise read future samples.
                    values = frame.to_ndarray()
                    count = allowed if frame.format.is_planar else allowed * len(frame.layout.channels)
                    clipped = av.AudioFrame.from_ndarray(values[:, :count].copy(), format=frame.format.name, layout=frame.layout.name)
                    clipped.sample_rate, clipped.pts, clipped.time_base = frame.sample_rate, frame.pts, frame.time_base
                    frame = clipped
            chunks.extend(resampler.resample(frame))
        chunks.extend(resampler.resample(None))
        if not chunks or any(frame.pts is None for frame in chunks):
            return None, 0., {"available": False, "reason": "missing_audio_pts"}
        origin = float(chunks[0].pts * chunks[0].time_base) - media_origin
        ends = [round((float(f.pts * f.time_base) - media_origin - origin) * sample_rate) + f.samples for f in chunks]
        waveform = torch.zeros(max(ends))
        previous_end = 0
        gaps = 0
        for frame, end in zip(chunks, ends):
            begin = end - frame.samples
            if begin < previous_end - 1:
                raise ValueError("overlapping/nonmonotonic audio PTS")
            if begin > previous_end + 1:
                gaps += 1
            values = torch.from_numpy(frame.to_ndarray().reshape(-1).copy())
            waveform[begin:end] = values
            previous_end = end
        return waveform, origin, {"available": True, "origin_s": origin, "pts_gaps": gaps,
                                   "causal_decode_end_s": end_s, "truncate_before_resampling": True,
                                   "duration_s": len(waveform) / sample_rate}


def frame_at_or_before(path, timestamp_s):
    """Decode the last presented video frame with PTS <= the decision time."""
    import av
    if not math.isfinite(timestamp_s) or timestamp_s < 0:
        raise ValueError("finite nonnegative video timestamp required")
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        origin = (container.start_time or 0) / av.time_base
        target = timestamp_s + origin
        container.seek(max(0, int(target / float(stream.time_base))), stream=stream, backward=True)
        chosen, pts = None, None
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("video frame has no presentation timestamp")
            current = float(frame.pts * frame.time_base) - origin
            if current > timestamp_s + 1e-8:
                break
            chosen, pts = frame.to_ndarray(format="rgb24"), current
        if chosen is None:
            raise ValueError("no causal frame before requested time")
        return chosen, pts
