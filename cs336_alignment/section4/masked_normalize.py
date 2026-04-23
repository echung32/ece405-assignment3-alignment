from __future__ import annotations

import torch


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """Sum masked tensor values and divide by a constant."""
    masked_tensor = tensor * mask.to(dtype=tensor.dtype)
    return masked_tensor.sum(dim=dim) / normalize_constant
