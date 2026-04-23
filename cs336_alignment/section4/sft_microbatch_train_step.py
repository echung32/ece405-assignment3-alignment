from __future__ import annotations

import torch

from cs336_alignment.section4.masked_normalize import masked_normalize


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Backpropagate the masked negative log-likelihood for one microbatch."""
    per_example_log_prob = masked_normalize(
        tensor=policy_log_probs,
        mask=response_mask,
        normalize_constant=normalize_constant,
        dim=1,
    )
    loss = -per_example_log_prob.mean()
    loss = loss / gradient_accumulation_steps
    loss.backward()
    metadata = {
        "mean_masked_log_prob": per_example_log_prob.detach().mean(),
        "num_response_tokens": response_mask.sum(),
    }
    return loss, metadata
