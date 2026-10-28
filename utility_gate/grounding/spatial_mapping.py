"""Exact positional support for SigLIP384 valid14 -> bilinear27to14.

Support is a weighted union of native patch cells, NOT a transformer receptive
field or a partition into equal 14x14 cells of the original image.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class ImageTransform:
    width: int
    height: int
    mode: str = 'warp'
    size: int = 384

    def affine(self):
        if min(self.width, self.height, self.size) <= 0:
            raise ValueError('positive image dimensions required')
        if self.mode == 'warp':
            return self.size/self.width, self.size/self.height, 0., 0.
        if self.mode not in ('center_crop', 'letterbox'):
            raise ValueError('unknown image transform')
        scale = (max if self.mode == 'center_crop' else min)(self.size/self.width, self.size/self.height)
        return scale, scale, (self.size-scale*self.width)/2, (self.size-scale*self.height)/2

    def boxes(self, boxes, normalized=True):
        a = np.asarray(boxes, dtype=np.float64)
        if not a.size:
            return np.empty((0, 4))
        if a.ndim != 2 or a.shape[1] != 4 or not np.isfinite(a).all():
            raise ValueError('finite xyxy boxes required')
        if np.any(a[:, 2:] < a[:, :2]):
            raise ValueError('inverted bbox')
        a = a.copy()
        if normalized:
            a *= [self.width, self.height, self.width, self.height]
        a[:, [0, 2]] = np.clip(a[:, [0, 2]], 0, self.width)
        a[:, [1, 3]] = np.clip(a[:, [1, 3]], 0, self.height)
        sx, sy, tx, ty = self.affine()
        a = a*[sx, sy, sx, sy]+[tx, ty, tx, ty]
        a = np.clip(a, 0, self.size)
        return a[np.all(a[:, 2:] > a[:, :2], axis=1)]


def union_area(boxes):
    """Continuous rectangle union, without double-counting overlaps."""
    if not len(boxes):
        return 0.
    xs = sorted({float(v) for b in boxes for v in (b[0], b[2])})
    area = 0.
    for lo, hi in zip(xs, xs[1:]):
        intervals = sorted((b[1], b[3]) for b in boxes if b[0] < hi and b[2] > lo)
        if not intervals:
            continue
        start, end = intervals[0]
        length = 0.
        for a, b in intervals[1:]:
            if a > end:
                length += end-start
                start, end = a, b
            else:
                end = max(end, b)
        area += (hi-lo)*(length+end-start)
    return float(area)


def native_occupancy(boxes, transform):
    """Fractional ROI union in each actual valid-convolution input patch."""
    boxes = transform.boxes(boxes)
    occupancy = np.zeros((27, 27), np.float64)
    for y in range(27):
        for x in range(27):
            cell = [x*14, y*14, (x+1)*14, (y+1)*14]
            clipped = [[max(cell[0], b[0]), max(cell[1], b[1]), min(cell[2], b[2]), min(cell[3], b[3])]
                       for b in boxes if min(cell[2], b[2]) > max(cell[0], b[0]) and min(cell[3], b[3]) > max(cell[1], b[1])]
            occupancy[y, x] = union_area(clipped)/196
    return occupancy


def pooled_occupancy(boxes, transform):
    native = torch.from_numpy(native_occupancy(boxes, transform))[None, None]
    return F.interpolate(native, size=(14, 14), mode='bilinear', align_corners=False).flatten().numpy()


def token_centers(transform):
    # Native patch center is14*(k+.5). align_corners=False sampling makes
    # output center27*(i+.5), hence the last6 input rows/columns are unused.
    coordinate = 27*(np.arange(14)+.5)
    x, y = np.meshgrid(coordinate, coordinate, indexing='xy')
    sx, sy, tx, ty = transform.affine()
    return np.stack(((x.flatten()-tx)/sx, (y.flatten()-ty)/sy), axis=1)


def routing_metrics(routing, occupancy, boxes, transform):
    routing = np.asarray(routing, np.float64)
    occupancy = np.asarray(occupancy, np.float64)
    if routing.shape != (64, 196) or occupancy.shape != (196,):
        raise ValueError('expected64x196 routing and196 occupancy')
    if not np.isfinite(routing).all() or not np.isfinite(occupancy).all():
        raise ValueError('nonfinite routing/ROI')
    if routing.min() < -1e-7 or not np.allclose(routing.sum(1), 1, atol=2e-5):
        raise ValueError('routing rows must be probability distributions')
    if occupancy.min() < -1e-8 or occupancy.max() > 1+1e-8:
        raise ValueError('invalid ROI fractions')
    s = routing.mean(0)
    mass, area = float(s@occupancy), float(occupancy.mean())
    center = token_centers(transform)[int(s.argmax())]
    pixels = np.asarray(boxes, float).reshape(-1, 4)*[transform.width, transform.height, transform.width, transform.height]
    pointing = any(b[0] <= center[0] < b[2] and b[1] <= center[1] < b[3] for b in pixels)
    return {'mass': mass, 'area_baseline': area, 'excess_mass': mass-area,
            'lift': mass/max(area, 1e-12), 'pointing': bool(pointing),
            'argmax_token': int(s.argmax()), 'pointing_original_xy': center.tolist()}


LAYOUT = {'input_hw': [384, 384], 'conv_kernel_stride': [14, 14], 'encoder_hw': [27, 27],
          'native_hw': [14, 14], 'order': 'row major: y*14+x', 'align_corners': False,
          'active_preprocess': 'RGB direct bicubic resize to384x384; no crop/letterbox',
          'roi_rule': 'exact continuous bbox union occupancy on27x27 valid patches, then actual bilinear weights',
          'unused_input_bottom_right_pixels': 6, 'scope': 'positional support, not global transformer receptive field',
          'pointing_rule': 'interpolation-weighted input patch center; half-open xyxy boxes; lowest-index argmax ties'}
