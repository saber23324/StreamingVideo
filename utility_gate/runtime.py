"""Frozen model adapter; the LLM receives visual values and ordinary text only."""
from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from .common import AUDIO_ROUTER, BEATS, REPO, VISION_ROUTER, fingerprint, sha256
from .features import feature_payload, prefix_features


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def clocked(function):
    synchronize()
    start = time.perf_counter()
    result = function()
    synchronize()
    return result, time.perf_counter() - start


def parse_prediction(text, dataset='streamingbench'):
    """Preserve the local strict letter scorer; retain parse failures as wrong."""
    text = text.strip()
    if dataset.startswith('mlvu') and ')' in text:
        index = text.index(')')
        text = text[index-1:index]
    return text if text in ('A', 'B', 'C', 'D') else None


def score_answer(generation, answer, dataset='streamingbench'):
    answer = str(answer).strip().upper()
    if answer not in 'ABCD' or len(answer) != 1:
        raise ValueError('gold answer must be one of the four option letters')
    logits = torch.tensor([generation['option_logits'][letter] for letter in 'ABCD'], dtype=torch.float64)
    if not torch.isfinite(logits).all():
        raise ValueError('nonfinite option logits')
    nll = float(-logits.log_softmax(0)['ABCD'.index(answer)])
    parsed = parse_prediction(generation['text'], dataset)
    full_nll = -generation['option_logprobs_full_vocab'][answer]
    if not np.isfinite([nll, full_nll]).all():
        raise ValueError('nonfinite gold NLL')
    return {**generation, 'prediction': parsed, 'parser_valid': parsed is not None,
            'correct': parsed == answer, 'gold_nll': nll, 'gold_nll_full_vocab': full_nll,
            'option_prediction': 'ABCD'[int(logits.argmax())],
            'nll_protocol': 'single gold option letter; first-token log-softmax normalized over A/B/C/D; full-vocabulary NLL retained'}


def load_router(path, visual_dim, device):
    from streamingtom.audio_bottleneck import AudioConditionedBottleneck
    payload = torch.load(path, map_location='cpu', weights_only=False)
    args = payload['args']
    router = AudioConditionedBottleneck(visual_dim=visual_dim, num_queries=int(args['num_queries']),
        hidden_size=int(args['bottleneck_hidden']), num_heads=int(args['num_heads']),
        stage=int(args.get('bottleneck_stage', 3)), latent_norm=args.get('latent_norm', 'none'),
        value_mode=args['value_mode'], architecture=args['architecture'],
        attention_temperature=float(args['attention_temperature']))
    missing, extra = router.load_state_dict(payload['audio_bottleneck'], strict=False)
    if set(missing) - {'latent_norm.weight'} or extra:
        raise ValueError('router checkpoint mismatch')
    if args['value_mode'] != 'native' or args['num_queries'] != 64:
        raise ValueError('requires the existing native-value K=64 baselines')
    return router.to(device=device, dtype=torch.float32).eval().requires_grad_(False)


class FrozenRuntime:
    def __init__(self, audio_router=AUDIO_ROUTER, vision_router=VISION_ROUTER, beats=BEATS):
        from relevance_gate.feature_cache import NativeExtractor
        from transformers import AutoTokenizer
        self.extractor = NativeExtractor(str(beats), local_tokens=True)
        self.model, self.device = self.extractor.model, self.extractor.device
        self.processor, self.audio_encoder = self.extractor.processor, self.extractor.audio
        self.tokenizer = AutoTokenizer.from_pretrained(self.extractor.metadata['pretrained'], local_files_only=True)
        self.audio_router = load_router(audio_router, self.model.config.hidden_size, self.device)
        self.vision_router = load_router(vision_router, self.model.config.hidden_size, self.device)
        if self.audio_router.stage != 3 or self.vision_router.stage != 1:
            raise ValueError('expected separately trained stage-3 audio and stage-1 static vision checkpoints')
        self.option_ids = [self.tokenizer.encode(letter, add_special_tokens=False) for letter in 'ABCD']
        if any(len(value) != 1 for value in self.option_ids):
            raise ValueError('single-token option letters required')
        self.option_ids = torch.tensor([x[0] for x in self.option_ids], device=self.device)
        self.waveforms = OrderedDict()
        self.strict_audio = OrderedDict()
        self.metadata = {'encoder': self.extractor.metadata, 'audio_router_sha256': sha256(audio_router),
            'vision_router_sha256': sha256(vision_router), 'K': 64,
            'scorer': 'local strict generated-letter accuracy, same QA manifest and prompt in every branch',
            'gold_nll': 'single-letter ABCD-normalized first-token NLL',
            'generation': {'do_sample': False, 'max_new_tokens': 4, 'use_cache': True},
            'features_version': 1, 'runtime_source_sha256': sha256(__file__),
            'legacy_audio_note': 'M0 prefix proxy preserves legacy whole-track resampling; M4 uses truncate-before-resampling causal audio',
            'source_sha256': {p: sha256(REPO / p) for p in ['streamingtom/audio_bottleneck/adbt.py',
                'streamingtom/audio_bottleneck/beats_encoder.py', 'streamingtom/tasks/vision.py',
                'streamingtom/train_adbt_videomme.py', 'streamingtom/models/llava/llava_streamingtom.py']}}
        if any(p.requires_grad for module in (self.model, self.audio_encoder, self.audio_router, self.vision_router)
               for p in module.parameters()):
            raise AssertionError('all baseline modules must be frozen')

    def prompt(self, row):
        from streamingtom.train_adbt_videomme import make_prompt
        return make_prompt(row, self.tokenizer)

    @torch.inference_mode()
    def generate(self, prompt_ids, latents):
        """One greedy VLM call, no gold argument; first logits also support offline NLL.

        Reuses the exact image-marker/embedding/position contract of the direct
        wrapper. Calling the native decoder exposes prefill/TTFT without a
        second teacher-forced VLM run. Correctness is scored afterwards.
        """
        from llava.constants import IMAGE_TOKEN_INDEX
        prompt_ids = prompt_ids.reshape(-1)
        markers = torch.where(prompt_ids == IMAGE_TOKEN_INDEX)[0]
        if markers.numel() != 1:
            raise ValueError('exactly one image marker required')
        marker = int(markers[0])
        embed = self.model.get_input_embeddings()
        device, dtype = embed.weight.device, embed.weight.dtype
        flattened = latents.reshape(-1, latents.shape[-1])
        synchronize()
        started = time.perf_counter()
        inputs = torch.cat((embed(prompt_ids[:marker].to(device))[None], flattened[None].to(device, dtype),
                            embed(prompt_ids[marker+1:].to(device))[None]), 1)
        length = inputs.shape[1]
        attention = torch.ones((1, length), device=device, dtype=torch.long)
        positions = torch.arange(length, device=device)[None]
        synchronize()
        prefill_start = time.perf_counter()
        output = self.model.model(inputs_embeds=inputs, attention_mask=attention,
                                  position_ids=positions, use_cache=True, return_dict=True)
        logits = self.model.lm_head(output.last_hidden_state[:, -1]).float()[0]
        synchronize()
        prefill_end = time.perf_counter()
        option_logits = logits[self.option_ids.to(logits.device)]
        logprobs = logits.log_softmax(0)[self.option_ids.to(logits.device)]
        generated = [int(logits.argmax())]
        first_token_time = time.perf_counter() - started
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos if eos is not None else self.tokenizer.eos_token_id])
        cache = output.past_key_values
        for step in range(1, 4):
            if generated[-1] in eos:
                break
            output = self.model.model(input_ids=torch.tensor([[generated[-1]]], device=device),
                attention_mask=torch.ones((1, length+step), device=device, dtype=torch.long),
                position_ids=torch.tensor([[length+step-1]], device=device),
                past_key_values=cache, use_cache=True, return_dict=True)
            cache = output.past_key_values
            logits = self.model.lm_head(output.last_hidden_state[:, -1]).float()[0]
            generated.append(int(logits.argmax()))
        synchronize()
        return {'text': self.tokenizer.decode(generated, skip_special_tokens=True).strip(),
            'generated_ids': generated, 'option_logits': dict(zip('ABCD', option_logits.cpu().tolist())),
            'option_logprobs_full_vocab': dict(zip('ABCD', logprobs.cpu().tolist())),
            'visual_tokens': len(flattened), 'total_input_tokens': length,
            'llm_prefill_s': prefill_end-prefill_start, 'llm_ttft_s': first_token_time,
            'llm_total_s': time.perf_counter()-started}

    def full_vision(self, visual):
        value = visual.reshape(-1, visual.shape[-1])
        config = self.model.config
        if (getattr(config, 'mm_patch_merge_type', 'flat').startswith('spatial')
                and 'unpad' in getattr(config, 'mm_patch_merge_type', '')
                and getattr(config, 'mm_newline_position', 'one_token') == 'one_token'):
            value = torch.cat((value, self.model.get_model().image_newline[None].to(value)))
        return value

    def causal_windows(self, path, timestamps):
        from relevance_gate.media import decode_audio, causal_window
        windows, valid, reasons, rms = [], [], [], []
        for stamp in timestamps:
            stamp = float(stamp)
            identity = (str(path), stamp)
            if identity not in self.strict_audio:
                try:
                    waveform, origin, details = decode_audio(path, end_s=stamp)
                    window = causal_window(waveform, stamp, origin_s=origin)
                    if details.get('pts_gaps', 0):
                        window.valid, window.reason = False, 'audio_pts_gap'
                    result = (window.samples, window.valid, window.reason, window.rms)
                except ValueError as error:
                    if str(error) != 'overlapping/nonmonotonic audio PTS':
                        raise
                    result = (torch.zeros(32000), False, 'audio_pts_overlap', 0.)
                self.strict_audio[identity] = result
                if len(self.strict_audio) > 64:
                    self.strict_audio.popitem(last=False)
            sample, okay, reason, energy = self.strict_audio[identity]
            windows.append(sample)
            valid.append(okay)
            reasons.append(reason)
            rms.append(energy)
        return torch.stack(windows), torch.tensor(valid, dtype=torch.bool), reasons, torch.tensor(rms)

    @torch.inference_mode()
    def encode(self, row, mode='prefix', max_frames=8):
        from streamingtom.train_adbt_videomme import sample_video_for_rows
        from relevance_gate.media import frame_at_or_before
        path = row['video_path']
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        if mode == 'prefix':
            (frames, timestamps, _), decode_s = clocked(lambda: sample_video_for_rows(path, [row], fps='auto', max_frames=max_frames))
            frame_pts = timestamps.tolist()
        else:
            end = float(row['query_time_seconds'])
            timestamps = np.arange(0., end + 1e-8, float(row.get('frame_interval_s', 2.)), dtype=np.float32)
            def decode():
                samples = [frame_at_or_before(path, float(t)) for t in timestamps]
                return np.stack([x[0] for x in samples]), [x[1] for x in samples]
            (frames, frame_pts), decode_s = clocked(decode)
        if row.get('query_time_seconds') is not None and (np.asarray(timestamps) > row['query_time_seconds']+1e-6).any():
            raise AssertionError('future frame')
        visual, visual_s = clocked(lambda: self.extractor.encode_visual(self.model, self.processor, frames, 4).float())
        (windows, valid, reasons, rms), audio_decode_s = clocked(lambda: self.causal_windows(path, timestamps))
        causal, audio_encoder_s = clocked(lambda: self.audio_encoder.encode_windows(windows, self.device).to(self.device).float())
        if mode == 'prefix':
            def legacy():
                if path not in self.waveforms:
                    self.waveforms[path] = self.audio_encoder.decode_audio(path)
                    if len(self.waveforms) > 2:
                        self.waveforms.popitem(last=False)
                waveform = self.waveforms[path]
                legacy_windows = self.audio_encoder._causal_windows(waveform, torch.as_tensor(timestamps))
                return self.audio_encoder.encode_windows(legacy_windows, self.device).to(self.device).float()
            audio, legacy_s = clocked(legacy)
        else:
            audio, legacy_s = causal, 0.
        times = torch.as_tensor(timestamps, device=self.device)
        (av, av_p), route_av_s = clocked(lambda: self.audio_router(visual, audio, times))
        (vision, vision_p), route_v_s = clocked(lambda: self.vision_router(visual, None, times))
        if visual.shape[1] != 196 or av.shape[1] != 64 or vision.shape != av.shape:
            raise AssertionError('token interface mismatch')
        if not torch.equal(av, av_p @ visual) or not torch.equal(vision, vision_p @ visual):
            raise AssertionError('native visual resampling constraint changed')
        features = prefix_features(visual, causal, valid, times)
        return {'visual': visual.cpu(), 'causal_audio': causal.cpu(), 'router_audio': audio.cpu(),
            'audio_latents': av.cpu(), 'vision_latents': vision.cpu(), 'timestamps': times.cpu(),
            'frame_pts_s': frame_pts, 'audio_valid': valid, 'audio_quality': reasons,
            'audio_rms': rms, 'audio_windows': windows, 'features': feature_payload(features),
            'mode': mode, 'timing': {'video_decode_s': decode_s, 'visual_encoder_s': visual_s,
                'causal_audio_decode_s': audio_decode_s, 'audio_encoder_s': audio_encoder_s,
                'legacy_audio_decode_and_encoder_s': legacy_s,
                'audio_routing_s': route_av_s, 'vision_routing_s': route_v_s,
                'two_branch_feature_total_s': time.perf_counter()-started,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved()}}

    @torch.inference_mode()
    def regression_fixture(self, fixture_dir):
        fixture = torch.load(Path(fixture_dir) / 'fixture.pt', map_location='cpu', weights_only=True)
        visual = torch.cat([chunk['visual'] for chunk in fixture['chunks']]).to(self.device)
        audio = fixture['audio_info']['embeddings'].to(self.device)
        z, p = self.audio_router(visual, audio, fixture['timestamps'].to(self.device))
        if not torch.equal(z.cpu(), torch.cat([c['latents'] for c in fixture['chunks']])):
            raise AssertionError('legacy routing no longer exactly matches captured fixture')
        result = self.generate(fixture['prompt_ids'], z)
        expected = fixture['generated_ids'].reshape(-1).tolist()
        if result['generated_ids'] != expected:
            raise AssertionError(f'native cached greedy decoding differs: {result["generated_ids"]} vs {expected}')
        return {'routing_exact': True, 'generated_tokens_exact': True, 'generated_ids': expected}
