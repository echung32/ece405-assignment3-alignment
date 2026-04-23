from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    """Tokenize prompt and response strings and align a response-only mask."""
    if len(prompt_strs) != len(output_strs):
        raise ValueError("prompt_strs and output_strs must have the same length")

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define a pad or eos token")

    tokenized_prompts = tokenizer(prompt_strs, add_special_tokens=False)["input_ids"]
    tokenized_outputs = tokenizer(output_strs, add_special_tokens=False)["input_ids"]

    combined_sequences: list[list[int]] = []
    response_masks: list[list[bool]] = []
    max_sequence_length = 0

    for prompt_ids, output_ids in zip(tokenized_prompts, tokenized_outputs, strict=True):
        combined_ids = list(prompt_ids) + list(output_ids)
        if len(combined_ids) < 2:
            raise ValueError("Each prompt/output pair must contain at least two tokens combined")

        combined_sequences.append(combined_ids)
        response_masks.append(
            [False] * max(len(prompt_ids) - 1, 0) + [True] * len(output_ids)
        )
        max_sequence_length = max(max_sequence_length, len(combined_ids))

    input_ids = []
    labels = []
    response_mask = []

    for combined_ids, mask in zip(combined_sequences, response_masks, strict=True):
        padded_ids = combined_ids + [pad_token_id] * (max_sequence_length - len(combined_ids))
        padded_mask = mask + [False] * (max_sequence_length - 1 - len(mask))
        input_ids.append(padded_ids[:-1])
        labels.append(padded_ids[1:])
        response_mask.append(padded_mask)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "response_mask": torch.tensor(response_mask, dtype=torch.bool),
    }
