from __future__ import annotations

from statistics import mean
from typing import Callable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from cs336_alignment.section4.get_response_log_probs import get_response_log_probs
from cs336_alignment.section4.tokenize_prompt_and_output import tokenize_prompt_and_output


def log_generations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    responses: list[str],
    ground_truths: list[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    device: torch.device,
    scoring_batch_size: int = 4,
) -> tuple[list[dict], dict[str, float | int]]:
    if not prompts:
        return [], {
            "num_logged_examples": 0,
            "mean_token_entropy": 0.0,
            "mean_response_length": 0.0,
            "mean_response_length_correct": 0.0,
            "mean_response_length_incorrect": 0.0,
        }

    was_training = model.training
    model.eval()
    avg_token_entropy_chunks: list[torch.Tensor] = []
    response_length_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start_idx in range(0, len(prompts), scoring_batch_size):
            end_idx = start_idx + scoring_batch_size
            tokenized = tokenize_prompt_and_output(
                prompt_strs=prompts[start_idx:end_idx],
                output_strs=responses[start_idx:end_idx],
                tokenizer=tokenizer,
            )
            input_ids = tokenized["input_ids"].to(device)
            labels = tokenized["labels"].to(device)
            response_mask = tokenized["response_mask"].to(device)
            scored = get_response_log_probs(
                model=model,
                input_ids=input_ids,
                labels=labels,
                response_mask=response_mask,
                return_token_entropy=True,
            )
            token_entropy = scored["token_entropy"]
            response_lengths = response_mask.sum(dim=1)
            entropy_denominator = response_lengths.clamp(min=1).to(dtype=token_entropy.dtype)
            avg_token_entropy = (
                (token_entropy * response_mask.to(dtype=token_entropy.dtype)).sum(dim=1) / entropy_denominator
            )
            avg_token_entropy_chunks.append(avg_token_entropy.cpu())
            response_length_chunks.append(response_lengths.cpu())
            del input_ids, labels, response_mask, scored, token_entropy, response_lengths, entropy_denominator, avg_token_entropy
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if was_training:
        model.train()

    avg_token_entropy = torch.cat(avg_token_entropy_chunks, dim=0)
    response_lengths = torch.cat(response_length_chunks, dim=0)

    entries: list[dict] = []
    correct_lengths: list[int] = []
    incorrect_lengths: list[int] = []

    for idx, (prompt, response, ground_truth) in enumerate(zip(prompts, responses, ground_truths, strict=True)):
        reward = reward_fn(response, ground_truth)
        response_length = int(response_lengths[idx].item())
        if reward["answer_reward"] == 1.0:
            correct_lengths.append(response_length)
        else:
            incorrect_lengths.append(response_length)
        entries.append(
            {
                "prompt": prompt,
                "response": response,
                "ground_truth": ground_truth,
                "reward": reward,
                "avg_token_entropy": float(avg_token_entropy[idx].item()),
                "response_length": response_length,
            }
        )

    summary = {
        "num_logged_examples": len(entries),
        "mean_token_entropy": mean(entry["avg_token_entropy"] for entry in entries),
        "mean_response_length": mean(entry["response_length"] for entry in entries),
        "mean_response_length_correct": mean(correct_lengths) if correct_lengths else 0.0,
        "mean_response_length_incorrect": mean(incorrect_lengths) if incorrect_lengths else 0.0,
    }
    return entries, summary
