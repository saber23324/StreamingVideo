"""Reuse frozen tensors after label/split review, verifying media identities."""

import json
from pathlib import Path

import torch

from .feature_cache import PairReader
from .manifest import cache_key, file_sha256, write_json


def reuse_cache_index(rows, source_index, output_index, manifest_path, encoder_kind=None):
    source_index, output_index = Path(source_index), Path(output_index)
    if output_index.exists():
        raise FileExistsError("choose a fresh cache index; existing indices are immutable")
    index = json.loads(source_index.read_text())
    metadata = index["encoder_metadata"]
    if encoder_kind is not None and metadata["kind"] != encoder_kind:
        raise ValueError("requested encoder differs from the cached encoder")
    by_key = {entry["key"]: entry for entry in index["entries"]}
    reader, entries = PairReader(), []
    for row in rows:
        # Labels and split membership do not affect encoder features. Media,
        # timestamps, corruption, source IDs and encoder identities still do.
        key = cache_key(row, metadata, reader.checksums(row))
        if key not in by_key:
            raise ValueError(f"media/time/encoder identity changed or features missing: {row['pair_id']}")
        entry = by_key[key]
        value = torch.load(entry["path"], map_location="cpu", weights_only=True)
        if value["cache_key"] != key or value["encoder_metadata"] != metadata:
            raise ValueError("cached tensor identity differs from its index")
        if any(not torch.isfinite(value[name]).all() or value[name].requires_grad for name in ("visual", "audio")):
            raise ValueError("reused encoder features must be finite and frozen")
        entries.append({"pair_id": row["pair_id"], "key": key, "path": str(Path(entry["path"]).resolve())})
    result = {"manifest_sha256": file_sha256(manifest_path), "encoder_metadata": metadata, "entries": entries,
              "reuse_provenance": {"source_index": str(source_index.resolve()), "source_index_sha256": file_sha256(source_index),
                                   "checks": "actual media hashes, feature identity, finite frozen tensors; no encoder executed"}}
    write_json(output_index, result)
    return result
