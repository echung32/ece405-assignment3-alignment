from __future__ import annotations

import torch
import torch.nn.functional as F

from cs336_alignment.section4.compute_entropy import compute_entropy


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Score next-token labels under a causal language model."""
    logits = model(input_ids).logits
    log_probs = F.log_softmax(logits.float(), dim=-1)
    token_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    outputs = {"log_probs": token_log_probs}
    if return_token_entropy:
        outputs["token_entropy"] = compute_entropy(logits)
    return outputs
