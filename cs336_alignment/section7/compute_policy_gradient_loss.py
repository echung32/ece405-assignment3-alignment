from __future__ import annotations

from typing import Literal

import torch

from cs336_alignment.section7.compute_grpo_clip_loss import compute_grpo_clip_loss
from cs336_alignment.section7.compute_grpo_no_clip_loss import compute_grpo_no_clip_loss
from cs336_alignment.section7.compute_naive_policy_gradient_loss import compute_naive_policy_gradient_loss


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == "no_baseline":
        if raw_rewards is None:
            raise ValueError("raw_rewards is required for no_baseline")
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}

    if loss_type == "reinforce_with_baseline":
        if advantages is None:
            raise ValueError("advantages is required for reinforce_with_baseline")
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}

    if loss_type == "grpo_clip":
        if advantages is None or old_log_probs is None or cliprange is None:
            raise ValueError("advantages, old_log_probs, and cliprange are required for grpo_clip")
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)

    if loss_type == "grpo_no_clip":
        if advantages is None or old_log_probs is None:
            raise ValueError("advantages and old_log_probs are required for grpo_no_clip")
        return compute_grpo_no_clip_loss(advantages, policy_log_probs, old_log_probs)

    raise ValueError(f"Unsupported loss_type: {loss_type}")