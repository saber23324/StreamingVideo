"""Verify official extracted frames against actual decoded video PTS."""
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image

from .manifest import file_sha256


def align_official_frames(video_path, frame_paths, pixel_mae_max=0.5, pixel_error_max=8):
    """Replay official 16-per-second sampling and verify supplied PNG pixels.

    Official scripts keep sample 15 of each 16, with integer rounded FPS.
    Filename index alone is never used as a timestamp. This is offline annotation
    alignment; gate feature extraction still receives only causal evidence.
    """
    frames, pts = [], []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("invalid official video FPS")
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("official video frame lacks PTS")
            pts.append(float(frame.pts * frame.time_base))
            frames.append(cv2.resize(frame.to_ndarray(format="rgb24"), (224, 224), interpolation=cv2.INTER_LINEAR))
    if not frames or any(right <= left for left, right in zip(pts, pts[1:])):
        raise ValueError("empty or nonmonotonic official video")
    rounded_fps = int(round(fps))
    records = []
    for index, frame_path in sorted(frame_paths.items()):
        with Image.open(frame_path) as image:
            expected = np.asarray(image.convert("RGB"))
        if expected.shape != (224, 224, 3):
            raise ValueError("official frame geometry differs from preprocessing recipe")
        decoded_index = min(int((int(index)-1) * rounded_fps + 15 / 16 * rounded_fps), len(frames)-1)
        difference = np.abs(frames[decoded_index].astype(np.int16) - expected.astype(np.int16))
        mae, maximum = float(difference.mean()), int(difference.max())
        if mae > pixel_mae_max or maximum > pixel_error_max:
            raise ValueError(f"official frame pixel check failed: {Path(frame_path).name} MAE={mae:.6f} max={maximum}")
        records.append({"frame_index": int(index), "timestamp_s": pts[decoded_index], "decoded_frame_index": decoded_index,
                        "frame_path": str(Path(frame_path).resolve()), "frame_sha256": file_sha256(frame_path),
                        "pixel_mae": mae, "pixel_max_error": maximum, "fps": fps,
                        "provenance": "official preprocess_s4/ms3.py 16-sample recipe; decoded PTS and PNG pixel verification"})
    return records
