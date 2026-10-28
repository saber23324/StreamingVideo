"""Strict official LLP TSV intervals; missing rows mean no annotated event."""
from __future__ import annotations

from collections import defaultdict
import csv
import re


def clip_identity(name):
    match = re.fullmatch(r'([\w-]{11})_(\d+)_(\d+)', name)
    if not match or int(match[3])-int(match[2]) != 10:
        raise ValueError('expected official YouTubeID_start_end10s')
    return match[1], int(match[2]), int(match[3])


def read_intervals(path, strict=True, rejections=None):
    rows = defaultdict(list)
    with open(path) as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        if reader.fieldnames != ['filename', 'onset', 'offset', 'event_labels']:
            raise ValueError('dense LLP TSV schema mismatch; weak labels are not intervals')
        for index, r in enumerate(reader, 2):
            try:
                clip_identity(r['filename'])
                start, end = float(r['onset']), float(r['offset'])
                if not start.is_integer() or not end.is_integer() or not 0 <= start < end <= 10:
                    raise ValueError('dense annotation must use integer half-open second boundaries')
                rows[r['filename']].append((int(start), int(end), r['event_labels']))
            except ValueError as error:
                if strict: raise
                if rejections is not None:
                    rejections.append({'line': index, 'path': str(path), 'reason': str(error), **r})
    return dict(rows)


def events(intervals, start, end):
    return {category for a, b, category in intervals if a < end and b > start}


def subtype(audio, visual):
    if audio & visual:
        return 'joint_same_class'
    if audio and visual:
        return 'different_categories'
    if audio:
        return 'audio_only'
    if visual:
        return 'visual_only'
    return 'neither'


def labels_at(audio_intervals, visual_intervals, time):
    if time != int(time) or not 1 <= time <= 10:
        raise ValueError('integer decision1..10 required')
    a, v = events(audio_intervals, time-1, time), events(visual_intervals, time-1, time)
    # Any SAME one-second bin has a class intersection; do not intersect unions
    # across two bins, which could invent a joint event at different times.
    bins = range(max(0, int(time)-2), int(time))
    joint = any(events(audio_intervals, k, k+1) & events(visual_intervals, k, k+1) for k in bins)
    return {'current_bin_joint': bool(a & v), 'causal_window_joint': bool(joint),
            'subtype': subtype(a, v), 'audio_categories': sorted(a), 'visual_categories': sorted(v),
            'joint_categories': sorted(a & v), 'current_bin': [time-1, time],
            'causal_label_window': [max(0, time-2), time]}
