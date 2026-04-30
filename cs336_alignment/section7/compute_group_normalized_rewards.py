from __future__ import annotations

from typing import Callable

import torch


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

    grouped_stds = grouped_rewards.std(dim=1, keepdim=True)
    if normalize_by_std:
        advantages = centered_rewards / (grouped_stds + advantage_eps)
    else:
        advantages = centered_rewards

    metadata = {
        "mean_raw_reward": raw_rewards.mean().item(),
        "std_raw_reward": raw_rewards.std().item(),
        "mean_group_std": grouped_stds.mean().item(),
        "mean_advantage": advantages.mean().item(),
    }
    return advantages.reshape(-1), raw_rewards, metadata