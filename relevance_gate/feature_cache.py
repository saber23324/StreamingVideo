"""Frozen native / ImageBind feature caches with media and encoder provenance."""

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import sys
import types

import numpy as np
from PIL import Image
import torch

from .manifest import cache_key, file_sha256, write_json
from .media import causal_window, decode_audio, frame_at_or_before


REPO = Path(__file__).resolve().parents[1]


def bind_legacy_imports():
    for directory in reversed([REPO, REPO / "LLaVA-NeXT", REPO / "lmms-eval"]):
        sys.path.insert(0, str(directory))


class NativeExtractor:
    def __init__(self, beats_checkpoint, pretrained="lmms-lab/llava-onevision-qwen2-7b-ov", local_tokens=False):
        bind_legacy_imports()
        from llava.mm_utils import get_model_name_from_path
        from llava.model.builder import load_pretrained_model
        from streamingtom.audio_bottleneck import BEATsAudioEncoder
        from streamingtom.train_adbt_videomme import encode_visual
        if not Path(beats_checkpoint).is_file():
            raise FileNotFoundError("real BEATs checkpoint required")
        _, self.model, self.processor, _ = load_pretrained_model(
            pretrained, None, get_model_name_from_path(pretrained), device_map="auto",
            multimodal=True, local_files_only=True, attn_implementation="flash_attention_2")
        self.model.eval().requires_grad_(False)
        self.encode_visual = encode_visual
        self.audio = BEATsAudioEncoder(str(beats_checkpoint), beats_root=str(REPO / "unilm/beats"),
                                       cache_dir="", batch_size=4).eval().requires_grad_(False)
        if self.audio.source != "beats":
            raise RuntimeError("diagnostic audio features cannot be used for gate experiments")
        self.device = next(self.model.get_vision_tower().parameters()).device
        self.local_tokens = local_tokens
        # Hash the actual projector values as well as the cached model revision.
        digest = hashlib.sha256()
        for name, tensor in self.model.get_model().mm_projector.state_dict().items():
            digest.update(name.encode()); digest.update(tensor.detach().cpu().numpy().tobytes())
        vision_digest = hashlib.sha256()
        for name, tensor in self.model.get_vision_tower().state_dict().items():
            vision_digest.update(name.encode())
            vision_digest.update(tensor.detach().cpu().numpy().tobytes())
        self.metadata = {"kind": "native", "pretrained": pretrained,
                         "model_commit": getattr(self.model.config, "_commit_hash", None),
                         "vision_state_sha256": vision_digest.hexdigest(),
                         "projector_sha256": digest.hexdigest(), "beats_sha256": file_sha256(beats_checkpoint),
                         "processor": {"class": type(self.processor).__name__, "size": list(self.processor.size),
                                       "image_mean": list(self.processor.image_mean), "image_std": list(self.processor.image_std),
                                       "resample": int(self.processor.resample), "rescale_factor": self.processor.rescale_factor,
                                       "applied_crop": False, "data_format": str(self.processor.data_format)},
                         "visual_pool": "tokens" if local_tokens else "mean",
                         "causal_audio_decode_version": 3,
                         "sample_rate": 16000, "audio_window_seconds": 2., "audio_states": "temporal",
                         "vision_source_sha256": file_sha256(REPO / "streamingtom/tasks/vision.py"),
                         "beats_source_sha256": file_sha256(REPO / "unilm/beats/BEATs.py")}

    @torch.inference_mode()
    def __call__(self, image, window):
        visual = self.encode_visual(self.model, self.processor, np.asarray(image)[None], 1)[0].float()
        if visual.shape != (196, self.model.config.hidden_size):
            raise ValueError("active native token layout differs from verified baseline")
        audio = self.audio.encode_windows(window.samples[None], self.device)[0] if window.valid else torch.zeros(1, 768)
        return {"visual": visual.cpu() if self.local_tokens else visual.mean(0, keepdim=True).cpu(),
                "audio": audio.float().cpu(), "original_visual_tokens": 196}


class ImageBindExtractor:
    def __init__(self, checkpoint, source=REPO / "external/ImageBind", device="cuda:0"):
        import torchvision.transforms as T
        if not Path(checkpoint).is_file():
            raise FileNotFoundError("download the official ImageBind weights first")
        source = Path(source).resolve()
        # Upstream __init__ eagerly imports an unused PyTorchVideo data module.
        # Load its unmodified model package as a namespace instead; the causal
        # image/audio preprocessing below does not use that video data loader.
        if "imagebind" not in sys.modules:
            namespace = types.ModuleType("imagebind")
            namespace.__path__ = [str(source / "imagebind")]
            sys.modules["imagebind"] = namespace
        elif str(source / "imagebind") not in list(sys.modules["imagebind"].__path__):
            raise ValueError("a different ImageBind source package is already loaded")
        from imagebind.models.imagebind_model import imagebind_huge, ModalityType
        self.modality = ModalityType
        self.model = imagebind_huge(pretrained=False)
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        # Unused modalities need not occupy GPU memory. Image/audio weights stay
        # exactly as released and never replace the native BEATs frontend.
        for modules in (self.model.modality_preprocessors, self.model.modality_trunks,
                        self.model.modality_heads, self.model.modality_postprocessors):
            for key in list(modules.keys()):
                if key not in (ModalityType.VISION, ModalityType.AUDIO):
                    del modules[key]
        self.model.eval().requires_grad_(False).to(device)
        self.device = device
        self.transform = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                                     T.CenterCrop(224), T.ToTensor(),
                                     T.Normalize((.48145466, .4578275, .40821073), (.26862954, .26130258, .27577711))])
        self.metadata = {"kind": "imagebind", "checkpoint_sha256": file_sha256(checkpoint),
                         "causal_audio_decode_version": 3,
                         "source_sha256": file_sha256(Path(source) / "imagebind/models/imagebind_model.py"),
                         "upstream_preprocessing_sha256": file_sha256(source / "imagebind/data.py"),
                         "adapter_source_sha256": file_sha256(Path(__file__)),
                         "import_scope": "unmodified official models; unused upstream video data loader not imported",
                         "vision_preprocessing": "official_resize224_centercrop_clip_normalization",
                         "audio_preprocessing": "kaldi_128mel_hanning_25ms_10ms_dither0_pad204_mean-4.268_std9.138",
                         "sample_rate": 16000, "audio_window_seconds": 2., "clips_per_window": 1,
                         "normalization": "cosine computed after L2 normalization of both embeddings"}

    @torch.inference_mode()
    def __call__(self, image, window):
        import torchaudio
        waveform = window.samples.clone()[None]
        waveform -= waveform.mean()
        mel = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True, sample_frequency=16000, use_energy=False,
            window_type="hanning", num_mel_bins=128, dither=0., frame_length=25, frame_shift=10).T
        mel = torch.nn.functional.pad(mel, (0, max(0, 204 - mel.shape[-1])))[:, :204]
        audio = ((mel[None] + 4.268) / 9.138)[None].to(self.device)
        output = self.model({self.modality.VISION: self.transform(image)[None].to(self.device),
                             self.modality.AUDIO: audio})
        return {"visual": output[self.modality.VISION].float().cpu(),
                "audio": output[self.modality.AUDIO].float().cpu(), "original_visual_tokens": None}


class PairReader:
    def __init__(self):
        self.audio_cache = OrderedDict()
        self.hashes = {}

    def checksums(self, row):
        result = {}
        for key in ("frame_path", "video_path", "audio_path", "source_region_mask"):
            path = row.get(key)
            if path:
                if path not in self.hashes:
                    self.hashes[path] = file_sha256(path)
                result[key] = self.hashes[path]
        return result

    def __call__(self, row):
        if row.get("frame_path"):
            with Image.open(row["frame_path"]) as source:
                image = source.convert("RGB")
            pts = row.get("frame_pts_s", row["frame_timestamp_s"])
        else:
            frame, pts = frame_at_or_before(row["video_path"], row["frame_timestamp_s"])
            image = Image.fromarray(frame)
        if pts > row["frame_timestamp_s"] + 1e-8:
            raise ValueError("future visual evidence")
        path = row.get("audio_path")
        audio_identity = (path, row["audio_end_s"])
        if audio_identity not in self.audio_cache:
            try:
                self.audio_cache[audio_identity] = decode_audio(path, end_s=row["audio_end_s"])
            except ValueError as error:
                if str(error) != "overlapping/nonmonotonic audio PTS":
                    raise
                self.audio_cache[audio_identity] = (None, 0., {"available": False, "reason": "audio_pts_overlap"})
            if len(self.audio_cache) > 4:
                self.audio_cache.popitem(last=False)
        waveform, origin, details = self.audio_cache[audio_identity]
        window = causal_window(waveform, row["audio_end_s"], origin_s=origin, start_s=row["audio_start_s"])
        if details.get("reason") == "audio_pts_overlap":
            window.valid, window.reason = False, "audio_pts_overlap"
        if details.get("pts_gaps", 0):
            window.valid, window.reason = False, "audio_pts_gap"
        # Older history remains causal but is unavailable as the current 2-s
        # auxiliary window; do not quietly revive stale media.
        if row["audio_start_s"] < max(0., row["frame_timestamp_s"] - 2.) - 1e-6:
            window.valid, window.reason = False, "out_of_window"
        corruption = row.get("corruption") or {}
        if corruption.get("type") == "silence":
            window.samples.zero_(); window.valid = False; window.reason = "silence"; window.rms = 0.
        elif corruption.get("type") == "noise":
            if window.valid:
                rng = torch.Generator().manual_seed(int(corruption["seed"]))
                noise = torch.randn(window.samples.shape, generator=rng)
                noise *= window.rms / (10 ** (float(corruption["snr_db"]) / 20))
                window.samples += noise * window.sample_mask
                window.rms = float(window.samples[window.sample_mask].square().mean().sqrt())
        elif corruption.get("type") not in (None, "none"):
            raise ValueError("unsupported corruption")
        return image, window, {**details, "frame_pts_s": float(pts)}


def build_cache(rows, extractor, cache_dir, index_path, manifest_path):
    cache_dir = Path(cache_dir)
    reader = PairReader()
    entries = []
    for index, row in enumerate(rows, 1):
        checksums = reader.checksums(row)
        key = cache_key(row, extractor.metadata, checksums)
        path = cache_dir / key[:2] / (key + ".pt")
        if not path.is_file():
            image, window, timing = reader(row)
            features = extractor(image, window)
            if not all(torch.isfinite(features[name]).all() for name in ("visual", "audio")):
                raise ValueError(f"nonfinite frozen features for {row['pair_id']}")
            features.update(valid_aux=window.valid, quality_reason=window.reason, rms=window.rms,
                            timing=timing, cache_key=key, encoder_metadata=extractor.metadata)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            torch.save(features, temporary); temporary.replace(path)
        entries.append({"pair_id": row["pair_id"], "path": str(path.resolve()), "key": key})
        if index % 25 == 0 or index == len(rows):
            print(f"cached {index}/{len(rows)} pairs", flush=True)
    result = {"manifest_sha256": file_sha256(manifest_path), "encoder_metadata": extractor.metadata, "entries": entries}
    write_json(index_path, result)
    return result


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, rows, index):
        paths = {r["pair_id"]: r for r in index["entries"]}
        if len(paths) != len(index["entries"]):
            raise ValueError("duplicate cache-index pair ID")
        self.rows = rows
        self.paths = paths
        if any(row["pair_id"] not in paths for row in rows):
            raise ValueError("feature cache is incomplete for this split")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        entry = self.paths[row["pair_id"]]
        value = torch.load(entry["path"], map_location="cpu", weights_only=True)
        if value["cache_key"] != entry["key"]:
            raise ValueError("feature/cache identity mismatch")
        return row, value


def collate_features(batch):
    from torch.nn.utils.rnn import pad_sequence
    rows, values = zip(*batch)
    result = {"rows": rows, "valid_aux": torch.tensor([v["valid_aux"] for v in values]),
              "quality_reasons": [v["quality_reason"] for v in values]}
    for name in ("visual", "audio"):
        tensors = [v[name].float() for v in values]
        result[name] = pad_sequence(tensors, batch_first=True)
        result[name + "_mask"] = pad_sequence([torch.ones(len(t), dtype=torch.bool) for t in tensors], batch_first=True)
    return result
