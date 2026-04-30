from __future__ import annotations

import torch


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    importance_ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = importance_ratio.clamp(min=1.0 - cliprange, max=1.0 + cliprange)
    unclipped_objective = importance_ratio * advantages
    clipped_objective = clipped_ratio * advantages
    clipped_is_active = clipped_objective < unclipped_objective
    loss = -torch.minimum(unclipped_objective, clipped_objective)
    metadata = {
        "importance_ratio": importance_ratio.detach(),
        "clipped_ratio": clipped_ratio.detach(),
        "is_clipped": clipped_is_active.detach(),
    }
    return loss, metadata