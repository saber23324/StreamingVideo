"""Default-off branch-switch policy, tested independently before VLM integration."""

import torch
from torch import nn

from .models import predict
from .types import RelevanceGateOutput


class RelevanceRoutingPolicy(nn.Module):
    def __init__(self, audio_router, vision_router=None, gate=None, calibration=None, enabled=False,
                 allow_provisional_calibration=False):
        super().__init__()
        self.audio_router = audio_router
        self.vision_router = vision_router
        self.gate = gate
        self.calibration = calibration
        self.enabled = enabled
        self.last_gate_output = None
        if enabled:
            if vision_router is None or gate is None or calibration is None:
                raise ValueError("enabled gate requires a validated gate, frozen calibration and explicit vision fallback")
            if calibration.get("label_protocol") == "weak_pairing" and not allow_provisional_calibration:
                raise ValueError("weak-pairing calibration requires explicit provisional development opt-in")
            if vision_router.stage != 1 or audio_router.stage != 3:
                raise ValueError("fallback must be the verified stage-1 strategy, audio branch stage-3")
            for name in ("num_queries", "visual_dim", "architecture", "value_mode", "latent_norm_type"):
                if getattr(audio_router, name) != getattr(vision_router, name):
                    raise ValueError(f"router token/interface mismatch: {name}")
            if audio_router.value_mode != "native":
                raise ValueError("gate policy preserves native visual resampling")
            for module in (audio_router, vision_router):
                if module.training or any(parameter.requires_grad for parameter in module.parameters()):
                    raise ValueError("legacy routers must already be frozen and in eval mode")
            gate.eval().requires_grad_(False)

    @property
    def architecture(self):
        return self.audio_router.architecture

    @property
    def num_queries(self):
        return self.audio_router.num_queries

    def forward(self, visual_tokens, audio_embeddings=None, end_timestamps=None,
                valid_aux=None, gate_visual=None, gate_audio=None):
        if not self.enabled:
            # Exact pass-through: no new casts, masks, state changes or gate calls.
            return self.audio_router(visual_tokens, audio_embeddings, end_timestamps)
        if valid_aux is None:
            raise ValueError("quality/time availability mask is mandatory")
        if valid_aux.dtype != torch.bool or valid_aux.shape != (len(visual_tokens),):
            raise ValueError("valid_aux must be boolean [B]")
        gate_visual = visual_tokens if gate_visual is None else gate_visual
        gate_audio = audio_embeddings if gate_audio is None else gate_audio
        if gate_audio is None or audio_embeddings is None:
            zeros = torch.zeros(len(visual_tokens), device=visual_tokens.device)
            rejected = torch.zeros(len(visual_tokens), dtype=torch.bool, device=visual_tokens.device)
            self.last_gate_output = RelevanceGateOutput(zeros, zeros, rejected, rejected,
                                                        diagnostics={"reason": "missing_audio_features", "score_kind": "unavailable"})
            return self.vision_router(torch.nan_to_num(visual_tokens), None, end_timestamps)
        result = predict(self.gate, gate_visual, gate_audio, valid_aux, self.calibration)
        self.last_gate_output = result
        if not result.accepted.any():
            return self.vision_router(torch.nan_to_num(visual_tokens), None, end_timestamps)
        if result.accepted.all():
            return self.audio_router(visual_tokens, audio_embeddings, end_timestamps)
        # Keep the ORIGINAL full timestamp batch for the audio branch. Selecting
        # accepted frames first would change the slot generator's delta-time.
        clean_visual = torch.where(torch.isfinite(visual_tokens), visual_tokens, 0.)
        clean_audio = torch.where(torch.isfinite(audio_embeddings), audio_embeddings, 0.)
        audio_z, audio_p = self.audio_router(clean_visual, clean_audio, end_timestamps)
        vision_z, vision_p = self.vision_router(clean_visual, None, end_timestamps)
        selected = result.accepted[:, None, None]
        return torch.where(selected, audio_z, vision_z), torch.where(selected, audio_p, vision_p)
