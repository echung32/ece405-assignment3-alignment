from __future__ import annotations

from typing import Callable, Literal

import torch

from cs336_alignment.section4.masked_normalize import masked_normalize


def compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError("rollout_responses and repeated_ground_truths must have the same length")
    if len(rollout_responses) % group_size != 0:
        raise ValueError("rollout batch size must be divisible by group_size")

    reward_outputs = [
        reward_fn(response, ground_truth)
        for response, ground_truth in zip(rollout_responses, repeated_ground_truths, strict=True)
    ]
    raw_rewards = torch.tensor([item["reward"] for item in reward_outputs], dtype=torch.float32)
    grouped_rewards = raw_rewards.reshape(-1, group_size)
    grouped_means = grouped_rewards.mean(dim=1, keepdim=True)
    centered_rewards = grouped_rewards - grouped_means

    if normalize_by_std:
        grouped_stds = grouped_rewards.std(dim=1, keepdim=True)
        advantages = centered_rewards / (grouped_stds + advantage_eps)
    else:
        grouped_stds = grouped_rewards.std(dim=1, keepdim=True)
        advantages = centered_rewards

    metadata = {
        "mean_raw_reward": raw_rewards.mean().item(),
        "std_raw_reward": raw_rewards.std().item(),
        "mean_group_std": grouped_stds.mean().item(),
        "mean_advantage": advantages.mean().item(),
    }
    return advantages.reshape(-1), raw_rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    return -(raw_rewards_or_advantages * policy_log_probs)


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


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    mask_as_tensor = mask.to(dtype=tensor.dtype)
    masked_tensor = tensor * mask_as_tensor
    if dim is None:
        denominator = mask_as_tensor.sum()
        return masked_tensor.sum() / denominator

    denominator = mask_as_tensor.sum(dim=dim)
    return masked_tensor.sum(dim=dim) / denominator


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