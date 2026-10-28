"""Cache frozen features for a validated correspondence manifest."""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from relevance_gate.feature_cache import NativeExtractor, ImageBindExtractor, build_cache
from relevance_gate.cache_reuse import reuse_cache_index
from relevance_gate.manifest import read_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--encoder", choices=("native", "imagebind"), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--beats-checkpoint", default="/nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt")
    parser.add_argument("--imagebind-checkpoint", default="/nvme_data/pkt/streaming/dataset/_models/imagebind/imagebind_huge.pth")
    parser.add_argument("--local-tokens", action="store_true")
    parser.add_argument("--reuse-index", type=Path, help="verify/reindex existing frozen features after label-only review; no GPU encoder")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    rows = read_manifest(args.manifest, check_files=True)
    if not rows:
        parser.error("manifest has no usable pairs")
    if args.reuse_index is not None:
        reuse_cache_index(rows, args.reuse_index, args.index, args.manifest, args.encoder)
        print(f"verified and reused {len(rows)} frozen feature pairs without running an encoder")
        return
    extractor = NativeExtractor(args.beats_checkpoint, local_tokens=args.local_tokens) if args.encoder == "native" else ImageBindExtractor(args.imagebind_checkpoint)
    build_cache(rows, extractor, args.cache_dir, args.index, args.manifest)


if __name__ == "__main__":
    main()
