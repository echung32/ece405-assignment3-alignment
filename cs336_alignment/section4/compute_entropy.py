from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute next-token entropy over the vocabulary dimension."""
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)
