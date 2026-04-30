from __future__ import annotations

import torch


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