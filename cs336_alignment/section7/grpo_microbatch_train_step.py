from __future__ import annotations

from typing import Literal

import torch

from cs336_alignment.section4.masked_normalize import masked_normalize
from cs336_alignment.section7.compute_policy_gradient_loss import compute_policy_gradient_loss
from cs336_alignment.section7.masked_mean import masked_mean


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    length_normalization: Literal["masked_mean", "masked_normalize"] = "masked_mean",
    normalize_constant: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    if length_normalization == "masked_mean":
        per_example_loss = masked_mean(per_token_loss, response_mask, dim=1)
    elif length_normalization == "masked_normalize":
        if normalize_constant is None:
            raise ValueError("normalize_constant is required for masked_normalize")
        per_example_loss = masked_normalize(
            tensor=per_token_loss,
            mask=response_mask,
            normalize_constant=normalize_constant,
            dim=1,
        )
    else:
        raise ValueError(f"Unsupported length_normalization: {length_normalization}")

    loss = per_example_loss.mean() / gradient_accumulation_steps
    loss.backward()

    detached_metadata: dict[str, torch.Tensor] = {
        "mean_per_example_loss": per_example_loss.detach().mean(),
        "num_response_tokens": response_mask.sum().detach(),
    }
    if "is_clipped" in metadata:
        detached_metadata["clip_fraction"] = masked_mean(
            metadata["is_clipped"].to(dtype=policy_log_probs.dtype),
            response_mask,
        ).detach()
    detached_metadata.update(metadata)
    return loss, detached_metadata