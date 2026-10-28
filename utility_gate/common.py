"""Small, deterministic IO and provenance helpers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = Path('/nvme_data/pkt/streaming/dataset')
AUDIO_ROUTER = REPO / 'results/adbt-phase4.1-q64-tau005-epoch3-lr1e5/adbt_epoch_3.pt'
VISION_ROUTER = REPO / 'results/exp01_visiononly64_train_epoch3_lr1e5/adbt_epoch_3.pt'
BEATS = Path('/nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def validate_splits(rows):
    """Group original identities, resolved media aliases and byte duplicates."""
    assigned, questions = {}, set()
    for row in rows:
        if row['split'] not in ('train', 'val', 'test'):
            raise ValueError('invalid split')
        if row['question_id'] in questions:
            raise ValueError('duplicate question ID')
        questions.add(row['question_id'])
        identities = [('source', row['source_video_id']), ('path', str(Path(row['video_path']).resolve()))]
        if row.get('media_sha256'):
            identities.append(('bytes', row['media_sha256']))
        for identity in identities:
            if identity in assigned and assigned[identity] != row['split']:
                raise ValueError('source video or duplicate media crosses split')
            assigned[identity] = row['split']
    return {'questions': len(rows), 'source_videos': len({r['source_video_id'] for r in rows}),
            'split_leakage': False, 'byte_duplicate_check': all('media_sha256' in r for r in rows)}
