from __future__ import annotations

import torch
import torch.nn.functional as F

from cs336_alignment.section4.compute_entropy import compute_entropy


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    response_mask: torch.Tensor | None = None,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Score next-token labels under a causal language model."""
    scoring_labels = labels
    logits_to_keep: int | torch.Tensor = 0
    left_pad = 0
    if response_mask is not None:
        max_response_length = int(response_mask.sum(dim=1).max().item())
        if max_response_length <= 0:
            raise ValueError("response_mask must include at least one response token")
        logits_to_keep = max_response_length
        scoring_labels = labels[:, -max_response_length:]
        left_pad = labels.shape[1] - max_response_length

    logits = model(input_ids, logits_to_keep=logits_to_keep).logits
    log_probs = F.log_softmax(logits.float(), dim=-1)
    token_log_probs = torch.gather(log_probs, dim=-1, index=scoring_labels.unsqueeze(-1)).squeeze(-1)
    if left_pad:
        token_log_probs = F.pad(token_log_probs, (left_pad, 0), value=0.0)

    outputs = {"log_probs": token_log_probs}
    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        if left_pad:
            token_entropy = F.pad(token_entropy, (left_pad, 0), value=0.0)
        outputs["token_entropy"] = token_entropy
    return outputs
