from __future__ import annotations

import torch


def compute_grpo_no_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    importance_ratio = torch.exp(policy_log_probs - old_log_probs)
    loss = -(importance_ratio * advantages)
    metadata = {
        "importance_ratio": importance_ratio.detach(),
    }
    return loss, metadata