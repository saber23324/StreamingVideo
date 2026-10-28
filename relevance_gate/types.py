from dataclasses import dataclass, field

import torch


@dataclass
class RelevanceGateOutput:
    match_logit: torch.Tensor
    match_score: torch.Tensor
    accepted: torch.Tensor
    valid_aux: torch.Tensor
    local_token_scores: torch.Tensor | None = None
    diagnostics: dict = field(default_factory=dict)
