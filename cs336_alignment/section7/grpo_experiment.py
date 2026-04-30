from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import wandb
from torch.nn.utils import clip_grad_norm_
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.section4.get_response_log_probs import get_response_log_probs
from cs336_alignment.section4.log_generations import log_generations
from cs336_alignment.section4.tokenize_prompt_and_output import tokenize_prompt_and_output
from cs336_alignment.section7.compute_group_normalized_rewards import compute_group_normalized_rewards
from cs336_alignment.section7.grpo_microbatch_train_step import grpo_microbatch_train_step
from cs336_alignment.section7.masked_mean import masked_mean

DEFAULT_MODEL_CANDIDATES = [Path("data/Qwen/Qwen2.5-Math-1.5B")]
DEFAULT_TRAIN_PATH = Path("data/math/train.jsonl")
DEFAULT_VAL_PATH = Path("data/math/val.jsonl")
DEFAULT_PROMPT_PATH = Path("cs336_alignment/prompts/r1_zero.prompt")
DEFAULT_LOG_ROOT = Path("logs/section7")
DEFAULT_OUTPUT_ROOT = Path("data/section7/grpo_experiment")
WANDB_ENTITY = "echung32-ece405"
WANDB_PROJECT = "ece405-alignment-grpo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Section 7 GRPO on MATH.")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val-path", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--n-grpo-steps", type=int, default=200)
    parser.add_argument("--rollout-batch-size", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--sampling-top-p", type=float, default=1.0)
    parser.add_argument("--sampling-min-tokens", type=int, default=4)
    parser.add_argument("--sampling-max-tokens", type=int, default=1024)
    parser.add_argument("--epochs-per-rollout-batch", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=128)
    parser.add_argument(
        "--loss-type",
        choices=["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"],
        default="reinforce_with_baseline",
    )
    parser.add_argument(
        "--reward-function",
        choices=["r1_zero", "question_only"],
        default="r1_zero",
    )
    parser.add_argument(
        "--length-normalization",
        choices=["masked_mean", "masked_normalize"],
        default="masked_mean",
    )
    parser.add_argument(
        "--length-normalize-constant",
        type=float,
        default=None,
        help="Constant denominator used when --length-normalization=masked_normalize. Defaults to sampling-max-tokens.",
    )
    parser.add_argument("--advantage-eps", type=float, default=1e-6)
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--use-std-normalization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-evals-per-run", type=int, default=10)
    parser.add_argument("--max-eval-examples", type=int, default=1024)
    parser.add_argument("--num-log-generations", type=int, default=32)
    parser.add_argument("--train-gpu-id", type=int, default=0)
    parser.add_argument("--eval-gpu-id", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--wandb-entity", type=str, default=WANDB_ENTITY)
    parser.add_argument("--wandb-project", type=str, default=WANDB_PROJECT)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--save-final-model", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def serialize_json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: serialize_json_value(nested_value) for key, nested_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_json_value(item) for item in value]
    return value


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory_stats(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def snapshot_cuda_memory(device: torch.device) -> dict[str, float | str]:
    snapshot: dict[str, float | str] = {"device": str(device)}
    if device.type != "cuda":
        return snapshot
    snapshot.update(
        {
            "allocated_mb": round(torch.cuda.memory_allocated(device) / (1024**2), 2),
            "reserved_mb": round(torch.cuda.memory_reserved(device) / (1024**2), 2),
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / (1024**2), 2),
            "peak_reserved_mb": round(torch.cuda.max_memory_reserved(device) / (1024**2), 2),
        }
    )
    return snapshot


def emit_step_trace(grpo_step: int, phase_timings: dict[str, float], snapshots: dict[str, dict[str, Any]]) -> None:
    trace_payload = {
        "grpo_step": grpo_step,
        "phase_seconds": {key: round(value, 3) for key, value in phase_timings.items()},
        "cuda_memory_mb": snapshots,
    }
    print(f"[step-trace] {json.dumps(trace_payload, sort_keys=True)}", flush=True)


def load_runtime_helpers() -> dict[str, Any]:
    from cs336_alignment.math_baseline import evaluate_vllm, summarize_metrics
    from cs336_alignment.section4.sft_experiment import (
        init_vllm,
        load_jsonl,
        load_policy_into_vllm_instance,
        load_policy_model,
        load_prompt_template,
        load_tokenizer,
        resolve_model_path,
        save_model_checkpoint,
        write_json,
        write_jsonl,
    )
    from cs336_alignment.section5.expert_iteration_experiment import release_memory

    return {
        "evaluate_vllm": evaluate_vllm,
        "init_vllm": init_vllm,
        "load_jsonl": load_jsonl,
        "load_policy_into_vllm_instance": load_policy_into_vllm_instance,
        "load_policy_model": load_policy_model,
        "load_prompt_template": load_prompt_template,
        "load_tokenizer": load_tokenizer,
        "release_memory": release_memory,
        "resolve_model_path": resolve_model_path,
        "save_model_checkpoint": save_model_checkpoint,
        "summarize_metrics": summarize_metrics,
        "write_json": write_json,
        "write_jsonl": write_jsonl,
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.train_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError("train_batch_size must be divisible by gradient_accumulation_steps")
    if args.rollout_batch_size % args.group_size != 0:
        raise ValueError("rollout_batch_size must be divisible by group_size")
    if args.train_batch_size < args.group_size:
        raise ValueError("train_batch_size must be greater than or equal to group_size")
    if args.loss_type == "grpo_clip" and args.epochs_per_rollout_batch < 1:
        raise ValueError("grpo_clip requires at least one epoch per rollout batch")
    if args.loss_type == "grpo_no_clip" and args.epochs_per_rollout_batch < 1:
        raise ValueError("grpo_no_clip requires at least one epoch per rollout batch")
    if args.length_normalization == "masked_normalize":
        normalize_constant = args.length_normalize_constant or float(args.sampling_max_tokens)
        if normalize_constant <= 0:
            raise ValueError("length_normalize_constant must be positive")


def build_prompt_examples(examples: list[dict[str, Any]], prompt_template: str) -> list[dict[str, Any]]:
    return [{**example, "prompt": prompt_template.format(question=example["problem"])} for example in examples]


def resolve_reward_fn(reward_function_name: str) -> Any:
    reward_fns = {
        "r1_zero": r1_zero_reward_fn,
        "question_only": question_only_reward_fn,
    }
    return reward_fns[reward_function_name]


def resolve_length_normalize_constant(args: argparse.Namespace) -> float | None:
    if args.length_normalization != "masked_normalize":
        return None
    if args.length_normalize_constant is not None:
        return float(args.length_normalize_constant)
    return float(args.sampling_max_tokens)


def build_eval_schedule(n_grpo_steps: int, num_evals_per_run: int) -> set[int]:
    schedule = {0, n_grpo_steps}
    for idx in range(1, num_evals_per_run + 1):
        schedule.add(max(1, math.ceil(n_grpo_steps * idx / num_evals_per_run)))
    return schedule


def build_rollout_sampling_params(args: argparse.Namespace) -> Any:
    from vllm import SamplingParams

    return SamplingParams(
        temperature=args.sampling_temperature,
        top_p=args.sampling_top_p,
        max_tokens=args.sampling_max_tokens,
        min_tokens=args.sampling_min_tokens,
        n=args.group_size,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=args.seed,
    )


def build_eval_sampling_params(args: argparse.Namespace) -> Any:
    from vllm import SamplingParams

    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.sampling_max_tokens,
        min_tokens=args.sampling_min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )


def sync_policy_to_vllm_if_needed(
    policy: PreTrainedModel,
    llm: Any,
    policy_version: int,
    synced_policy_version: int | None,
    runtime: dict[str, Any],
    *devices: torch.device,
) -> int:
    if synced_policy_version == policy_version:
        return policy_version
    runtime["load_policy_into_vllm_instance"](policy, llm)
    runtime["release_memory"](*devices)
    return policy_version


def run_eval(
    policy: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    eval_examples: list[dict[str, Any]],
    sampling_params: Any,
    train_device: torch.device,
    llm: Any,
    num_log_generations: int,
    reward_fn: Any,
    runtime: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prompts = [example["prompt"] for example in eval_examples]
    ground_truths = [example["solution"] for example in eval_examples]
    responses, metrics_list = runtime["evaluate_vllm"](
        vllm_model=llm,
        reward_fn=reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampling_params,
    )
    summary = runtime["summarize_metrics"](metrics_list)
    generation_logs, generation_summary = log_generations(
        model=policy,
        tokenizer=tokenizer,
        prompts=prompts[:num_log_generations],
        responses=responses[:num_log_generations],
        ground_truths=ground_truths[:num_log_generations],
        reward_fn=reward_fn,
        device=train_device,
    )
    for log_entry, example in zip(generation_logs, eval_examples[:num_log_generations], strict=True):
        log_entry["problem"] = example["problem"]
        log_entry["level"] = example.get("level")
        log_entry["type"] = example.get("type")
    summary.update(generation_summary)
    runtime["release_memory"](train_device)
    return summary, generation_logs


def format_float_tag(value: float) -> str:
    return f"{value:.0e}".replace("-", "m").replace("+", "")


def make_run_config_hash(args: argparse.Namespace) -> str:
    hash_payload = {
        "advantage_eps": args.advantage_eps,
        "cliprange": args.cliprange,
        "epochs_per_rollout_batch": args.epochs_per_rollout_batch,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "group_size": args.group_size,
        "learning_rate": args.learning_rate,
        "length_normalization": args.length_normalization,
        "length_normalize_constant": resolve_length_normalize_constant(args),
        "loss_type": args.loss_type,
        "model_path": args.model_path,
        "n_grpo_steps": args.n_grpo_steps,
        "prompt_path": str(args.prompt_path),
        "reward_function": args.reward_function,
        "rollout_batch_size": args.rollout_batch_size,
        "sampling_max_tokens": args.sampling_max_tokens,
        "sampling_min_tokens": args.sampling_min_tokens,
        "sampling_temperature": args.sampling_temperature,
        "sampling_top_p": args.sampling_top_p,
        "seed": args.seed,
        "train_batch_size": args.train_batch_size,
        "use_std_normalization": args.use_std_normalization,
        "weight_decay": args.weight_decay,
    }
    serialized_payload = json.dumps(hash_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(serialized_payload.encode("ascii")).hexdigest()[:8]


def make_run_name(args: argparse.Namespace) -> str:
    std_tag = "std" if args.use_std_normalization else "mean"
    run_name = (
        f"lr_{format_float_tag(args.learning_rate)}_"
        f"loss_{args.loss_type}_"
        f"{std_tag}_"
        f"g{args.group_size}_"
        f"rb{args.rollout_batch_size}_"
        f"ep{args.epochs_per_rollout_batch}"
    )
    if args.length_normalization == "masked_normalize":
        normalize_constant = resolve_length_normalize_constant(args)
        if normalize_constant is not None:
            normalize_tag = int(normalize_constant) if float(normalize_constant).is_integer() else normalize_constant
            run_name += f"_lnorm_const{normalize_tag}"
    if args.reward_function != "r1_zero":
        run_name += f"_reward_{args.reward_function}"
    return f"{run_name}_cfg{make_run_config_hash(args)}"


def sample_rollout_examples(
    train_examples: list[dict[str, Any]],
    prompts_per_batch: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    sample_size = min(prompts_per_batch, len(train_examples))
    if sample_size == len(train_examples):
        return list(train_examples)
    return rng.sample(train_examples, sample_size)


def generate_grouped_responses(llm: Any, prompts: list[str], sampling_params: Any) -> list[list[str]]:
    outputs = llm.generate(prompts, sampling_params)
    return [[candidate.text for candidate in output.outputs] for output in outputs]


def build_rollout_records(
    sampled_examples: list[dict[str, Any]],
    grouped_responses: list[list[str]],
    reward_fn: Any,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for example, responses in zip(sampled_examples, grouped_responses, strict=True):
        for rollout_index, response in enumerate(responses):
            metrics = reward_fn(response, example["solution"])
            records.append(
                {
                    "problem": example["problem"],
                    "solution": example["solution"],
                    "prompt": example["prompt"],
                    "response": response,
                    "rollout_index": rollout_index,
                    "metrics": metrics,
                }
            )
    return records


def summarize_rollout_records(records: list[dict[str, Any]], advantages: torch.Tensor) -> dict[str, Any]:
    if not records:
        return {
            "num_rollouts": 0,
            "num_correct_rollouts": 0,
            "mean_reward": 0.0,
            "mean_answer_reward": 0.0,
            "mean_format_reward": 0.0,
            "mean_advantage": 0.0,
            "mean_response_length": 0.0,
        }
    response_lengths = [len(record["response"]) for record in records]
    return {
        "num_rollouts": len(records),
        "num_correct_rollouts": sum(1 for record in records if record["metrics"]["answer_reward"] == 1.0),
        "mean_reward": sum(record["metrics"]["reward"] for record in records) / len(records),
        "mean_answer_reward": sum(record["metrics"]["answer_reward"] for record in records) / len(records),
        "mean_format_reward": sum(record["metrics"]["format_reward"] for record in records) / len(records),
        "mean_advantage": float(advantages.mean().item()),
        "mean_response_length": sum(response_lengths) / len(response_lengths),
    }


def run_grpo_epoch_chunk(
    policy: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    rollout_records: list[dict[str, Any]],
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor | None,
    train_device: torch.device,
    train_batch_size: int,
    gradient_accumulation_steps: int,
    loss_type: str,
    cliprange: float,
    length_normalization: str,
    normalize_constant: float | None,
    rng: random.Random,
    train_step_offset: int,
    runtime: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    if not rollout_records:
        return [], train_step_offset

    policy.train()
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    prompts = [record["prompt"] for record in rollout_records]
    responses = [record["response"] for record in rollout_records]
    tokenized = tokenize_prompt_and_output(prompts, responses, tokenizer)
    input_ids_cpu = tokenized["input_ids"]
    labels_cpu = tokenized["labels"]
    response_mask_cpu = tokenized["response_mask"]
    raw_rewards_cpu = raw_rewards.unsqueeze(1)
    advantages_cpu = advantages.unsqueeze(1)

    indices = list(range(len(rollout_records)))
    rng.shuffle(indices)
    train_history: list[dict[str, Any]] = []
    train_step = train_step_offset
    for batch_start in range(0, len(indices), train_batch_size):
        minibatch_indices = indices[batch_start : batch_start + train_batch_size]
        microbatch_chunks = [
            minibatch_indices[start : start + micro_train_batch_size]
            for start in range(0, len(minibatch_indices), micro_train_batch_size)
        ]
        optimizer.zero_grad(set_to_none=True)
        chunk_losses: list[float] = []
        chunk_entropies: list[float] = []
        chunk_rewards: list[float] = []
        chunk_answer_rewards: list[float] = []
        chunk_format_rewards: list[float] = []
        chunk_clip_fractions: list[float] = []
        chunk_tokens = 0

        for microbatch_indices in microbatch_chunks:
            tensor_indices = torch.tensor(microbatch_indices, dtype=torch.long)
            input_ids = input_ids_cpu.index_select(0, tensor_indices).to(train_device)
            labels = labels_cpu.index_select(0, tensor_indices).to(train_device)
            response_mask = response_mask_cpu.index_select(0, tensor_indices).to(train_device)
            scored = get_response_log_probs(
                model=policy,
                input_ids=input_ids,
                labels=labels,
                response_mask=response_mask,
                return_token_entropy=True,
            )
            micro_old_log_probs = None
            if old_log_probs is not None:
                micro_old_log_probs = old_log_probs.index_select(0, tensor_indices).to(train_device)
            micro_raw_rewards = raw_rewards_cpu.index_select(0, tensor_indices).to(train_device)
            micro_advantages = advantages_cpu.index_select(0, tensor_indices).to(train_device)

            loss, metadata = grpo_microbatch_train_step(
                policy_log_probs=scored["log_probs"],
                response_mask=response_mask,
                gradient_accumulation_steps=len(microbatch_chunks),
                loss_type=loss_type,
                raw_rewards=micro_raw_rewards,
                advantages=micro_advantages,
                old_log_probs=micro_old_log_probs,
                cliprange=cliprange,
                length_normalization=length_normalization,
                normalize_constant=normalize_constant,
            )
            token_entropy = masked_mean(scored["token_entropy"], response_mask).detach()
            chunk_losses.append(float(loss.detach().item()))
            chunk_entropies.append(float(token_entropy.item()))
            chunk_rewards.append(float(micro_raw_rewards.mean().item()))
            chunk_tokens += int(response_mask.sum().item())

            for record_index in microbatch_indices:
                record = rollout_records[record_index]
                chunk_answer_rewards.append(float(record["metrics"]["answer_reward"]))
                chunk_format_rewards.append(float(record["metrics"]["format_reward"]))
            if "clip_fraction" in metadata:
                chunk_clip_fractions.append(float(metadata["clip_fraction"].item()))

            del metadata, loss, scored, token_entropy, input_ids, labels, response_mask, micro_raw_rewards, micro_advantages
            if micro_old_log_probs is not None:
                del micro_old_log_probs

        runtime["release_memory"](train_device)
        grad_norm = float(clip_grad_norm_(policy.parameters(), max_norm=1.0).item())
        optimizer.step()
        runtime["release_memory"](train_device)
        train_step += 1
        train_record = {
            "train_step": train_step,
            "loss": sum(chunk_losses),
            "grad_norm": grad_norm,
            "mean_token_entropy": sum(chunk_entropies) / len(chunk_entropies),
            "mean_reward": sum(chunk_rewards) / len(chunk_rewards),
            "mean_answer_reward": sum(chunk_answer_rewards) / len(chunk_answer_rewards),
            "mean_format_reward": sum(chunk_format_rewards) / len(chunk_format_rewards),
            "response_tokens": chunk_tokens,
        }
        if chunk_clip_fractions:
            train_record["clip_fraction"] = sum(chunk_clip_fractions) / len(chunk_clip_fractions)
        train_history.append(train_record)

    return train_history, train_step


def cache_old_log_probs_chunked(
    policy: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    rollout_records: list[dict[str, Any]],
    train_device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    prompts = [record["prompt"] for record in rollout_records]
    responses = [record["response"] for record in rollout_records]
    tokenized = tokenize_prompt_and_output(prompts, responses, tokenizer)
    input_ids_cpu = tokenized["input_ids"]
    labels_cpu = tokenized["labels"]
    response_mask_cpu = tokenized["response_mask"]

    cached_chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for chunk_start in range(0, len(rollout_records), chunk_size):
            chunk_end = chunk_start + chunk_size
            input_ids = input_ids_cpu[chunk_start:chunk_end].to(train_device)
            labels = labels_cpu[chunk_start:chunk_end].to(train_device)
            response_mask = response_mask_cpu[chunk_start:chunk_end].to(train_device)
            old_scored = get_response_log_probs(
                model=policy,
                input_ids=input_ids,
                labels=labels,
                response_mask=response_mask,
                return_token_entropy=False,
            )
            cached_chunks.append(old_scored["log_probs"].detach().cpu())
            del old_scored, input_ids, labels, response_mask

    del tokenized, input_ids_cpu, labels_cpu, response_mask_cpu
    return torch.cat(cached_chunks, dim=0)


def train_grpo(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    run_start_time = time.perf_counter()
    runtime = load_runtime_helpers()
    reward_fn = resolve_reward_fn(args.reward_function)
    length_normalize_constant = resolve_length_normalize_constant(args)

    model_path = runtime["resolve_model_path"](args.model_path)
    tokenizer = runtime["load_tokenizer"](model_path)
    prompt_template = runtime["load_prompt_template"](args.prompt_path)
    train_examples = build_prompt_examples(runtime["load_jsonl"](args.train_path), prompt_template)
    eval_examples = build_prompt_examples(runtime["load_jsonl"](args.val_path), prompt_template)
    if args.max_eval_examples is not None:
        eval_examples = eval_examples[: args.max_eval_examples]

    run_name = make_run_name(args)
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)
    runtime["write_json"](output_dir / "config.json", serialize_json_value(vars(args)))

    train_device = torch.device(f"cuda:{args.train_gpu_id}" if torch.cuda.is_available() else "cpu")
    eval_device = torch.device(f"cuda:{args.eval_gpu_id}" if torch.cuda.is_available() else "cpu")
    policy = runtime["load_policy_model"](model_path=model_path, train_device=train_device)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    llm = runtime["init_vllm"](
        model_id=str(model_path),
        eval_gpu_id=args.eval_gpu_id,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    rollout_sampling_params = build_rollout_sampling_params(args)
    eval_sampling_params = build_eval_sampling_params(args)
    eval_schedule = build_eval_schedule(args.n_grpo_steps, args.num_evals_per_run)

    if args.wandb_group is None:
        args.wandb_group = f"section7-grpo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    wandb_run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=args.wandb_group,
        name=run_name,
        config={
            "run_name": run_name,
            "model_path": str(model_path),
            "num_train_examples": len(train_examples),
            "num_eval_examples": len(eval_examples),
            "learning_rate": args.learning_rate,
            "n_grpo_steps": args.n_grpo_steps,
            "rollout_batch_size": args.rollout_batch_size,
            "group_size": args.group_size,
            "epochs_per_rollout_batch": args.epochs_per_rollout_batch,
            "train_batch_size": args.train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "loss_type": args.loss_type,
            "reward_function": args.reward_function,
            "length_normalization": args.length_normalization,
            "length_normalize_constant": length_normalize_constant,
            "use_std_normalization": args.use_std_normalization,
        },
    )
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("grpo_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")
    wandb.define_metric("grpo/*", step_metric="grpo_step")

    train_rng = random.Random(args.seed)
    train_history: list[dict[str, Any]] = []
    eval_history: list[dict[str, Any]] = []
    step_summaries: list[dict[str, Any]] = []
    generation_artifacts: list[dict[str, Any]] = []
    best_eval_record: dict[str, Any] | None = None
    best_model_dir = output_dir / "best_model"
    train_step = 0
    synced_policy_version: int | None = None

    synced_policy_version = sync_policy_to_vllm_if_needed(
        policy,
        llm,
        train_step,
        synced_policy_version,
        runtime,
        train_device,
        eval_device,
    )
    initial_eval_summary, initial_generation_logs = run_eval(
        policy=policy,
        tokenizer=tokenizer,
        eval_examples=eval_examples,
        sampling_params=eval_sampling_params,
        train_device=train_device,
        llm=llm,
        num_log_generations=args.num_log_generations,
        reward_fn=reward_fn,
        runtime=runtime,
    )
    initial_eval_record = {
        "eval_step": 0,
        "grpo_step": 0,
        "elapsed_seconds": time.perf_counter() - run_start_time,
        **initial_eval_summary,
    }
    eval_history.append(initial_eval_record)
    generation_artifacts.append({"eval_step": 0, "logs": initial_generation_logs})
    best_eval_record = dict(initial_eval_record)
    if args.save_final_model:
        runtime["save_model_checkpoint"](policy=policy, tokenizer=tokenizer, output_dir=best_model_dir)
        runtime["release_memory"](train_device, eval_device)
    wandb.log(
        {
            "eval_step": 0,
            "grpo_step": 0,
            **{f"eval/{key}": value for key, value in initial_eval_summary.items() if isinstance(value, (int, float))},
        }
    )

    prompts_per_batch = args.rollout_batch_size // args.group_size
    for grpo_step in range(1, args.n_grpo_steps + 1):
        step_started_at = time.perf_counter()
        phase_timings: dict[str, float] = {}
        train_memory_snapshots: dict[str, dict[str, Any]] = {}
        reset_peak_memory_stats(train_device)

        sampled_examples = sample_rollout_examples(train_examples, prompts_per_batch, train_rng)
        prompts = [example["prompt"] for example in sampled_examples]
        phase_started_at = time.perf_counter()
        synced_policy_version = sync_policy_to_vllm_if_needed(
            policy,
            llm,
            train_step,
            synced_policy_version,
            runtime,
            train_device,
            eval_device,
        )
        synchronize_device(train_device)
        phase_timings["sync_policy"] = time.perf_counter() - phase_started_at
        train_memory_snapshots["after_sync_policy"] = snapshot_cuda_memory(train_device)

        phase_started_at = time.perf_counter()
        grouped_responses = generate_grouped_responses(llm, prompts, rollout_sampling_params)
        phase_timings["rollout_generate"] = time.perf_counter() - phase_started_at
        train_memory_snapshots["after_rollout_generate"] = snapshot_cuda_memory(train_device)

        phase_started_at = time.perf_counter()
        rollout_records = build_rollout_records(sampled_examples, grouped_responses, reward_fn=reward_fn)
        rollout_responses = [record["response"] for record in rollout_records]
        repeated_ground_truths = [record["solution"] for record in rollout_records]
        advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            reward_fn=reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=args.group_size,
            advantage_eps=args.advantage_eps,
            normalize_by_std=args.use_std_normalization,
        )
        for record, advantage, raw_reward in zip(rollout_records, advantages.tolist(), raw_rewards.tolist(), strict=True):
            record["advantage"] = advantage
            record["raw_reward"] = raw_reward
        phase_timings["reward_compute"] = time.perf_counter() - phase_started_at
        train_memory_snapshots["after_reward_compute"] = snapshot_cuda_memory(train_device)

        phase_started_at = time.perf_counter()
        step_dir = output_dir / f"step_{grpo_step:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        runtime["write_jsonl"](step_dir / "rollouts.jsonl", rollout_records)
        phase_timings["artifact_write"] = time.perf_counter() - phase_started_at

        cached_old_log_probs = None
        phase_started_at = time.perf_counter()
        if args.loss_type in {"grpo_clip", "grpo_no_clip"}:
            old_log_prob_chunk_size = max(1, min(args.train_batch_size, 16))
            cached_old_log_probs = cache_old_log_probs_chunked(
                policy=policy,
                tokenizer=tokenizer,
                rollout_records=rollout_records,
                train_device=train_device,
                chunk_size=old_log_prob_chunk_size,
            )
            runtime["release_memory"](train_device)
        synchronize_device(train_device)
        phase_timings["cache_old_log_probs"] = time.perf_counter() - phase_started_at
        train_memory_snapshots["after_cache_old_log_probs"] = snapshot_cuda_memory(train_device)

        step_train_history: list[dict[str, Any]] = []
        phase_started_at = time.perf_counter()
        for epoch_index in range(args.epochs_per_rollout_batch):
            epoch_history, train_step = run_grpo_epoch_chunk(
                policy=policy,
                tokenizer=tokenizer,
                optimizer=optimizer,
                rollout_records=rollout_records,
                raw_rewards=raw_rewards,
                advantages=advantages,
                old_log_probs=cached_old_log_probs,
                train_device=train_device,
                train_batch_size=args.train_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                loss_type=args.loss_type,
                cliprange=args.cliprange,
                length_normalization=args.length_normalization,
                normalize_constant=length_normalize_constant,
                rng=train_rng,
                train_step_offset=train_step,
                runtime=runtime,
            )
            for record in epoch_history:
                record["grpo_step"] = grpo_step
                record["epoch_within_rollout_batch"] = epoch_index + 1
                record["elapsed_seconds"] = time.perf_counter() - run_start_time
            step_train_history.extend(epoch_history)
            train_history.extend(epoch_history)
            for record in epoch_history:
                numeric_train_values = {
                    f"train/{key}": value
                    for key, value in record.items()
                    if key not in {"train_step", "grpo_step", "epoch_within_rollout_batch"} and isinstance(value, (int, float))
                }
                wandb.log({"train_step": record["train_step"], "grpo_step": grpo_step, **numeric_train_values})

        synchronize_device(train_device)
        phase_timings["train_update"] = time.perf_counter() - phase_started_at
        train_memory_snapshots["after_train_update"] = snapshot_cuda_memory(train_device)

        step_summary = summarize_rollout_records(rollout_records, advantages)
        step_summary.update(reward_metadata)
        step_summary["grpo_step"] = grpo_step
        step_summary["train_updates"] = len(step_train_history)
        step_summary["elapsed_seconds"] = time.perf_counter() - run_start_time
        step_summary["phase_seconds"] = {}
        step_summary["train_cuda_memory_mb"] = {}
        step_summaries.append(step_summary)
        runtime["write_json"](step_dir / "summary.json", step_summary)
        runtime["write_jsonl"](step_dir / "train_history.jsonl", step_train_history)
        wandb.log(
            {
                "grpo_step": grpo_step,
                **{f"grpo/{key}": value for key, value in step_summary.items() if isinstance(value, (int, float))},
            }
        )

        eval_sync_seconds = 0.0
        eval_seconds = 0.0
        if grpo_step in eval_schedule:
            phase_started_at = time.perf_counter()
            synced_policy_version = sync_policy_to_vllm_if_needed(
                policy,
                llm,
                train_step,
                synced_policy_version,
                runtime,
                train_device,
                eval_device,
            )
            synchronize_device(train_device)
            eval_sync_seconds = time.perf_counter() - phase_started_at
            train_memory_snapshots["after_eval_sync_policy"] = snapshot_cuda_memory(train_device)

            phase_started_at = time.perf_counter()
            eval_summary, generation_logs = run_eval(
                policy=policy,
                tokenizer=tokenizer,
                eval_examples=eval_examples,
                sampling_params=eval_sampling_params,
                train_device=train_device,
                llm=llm,
                num_log_generations=args.num_log_generations,
                reward_fn=reward_fn,
                runtime=runtime,
            )
            synchronize_device(train_device)
            eval_seconds = time.perf_counter() - phase_started_at
            train_memory_snapshots["after_eval"] = snapshot_cuda_memory(train_device)
            eval_record = {
                "eval_step": grpo_step,
                "grpo_step": grpo_step,
                "elapsed_seconds": time.perf_counter() - run_start_time,
                **eval_summary,
            }
            eval_history.append(eval_record)
            generation_artifacts.append({"eval_step": grpo_step, "logs": generation_logs})
            if best_eval_record is None or eval_record["accuracy"] > best_eval_record["accuracy"]:
                best_eval_record = dict(eval_record)
                if args.save_final_model:
                    runtime["save_model_checkpoint"](policy=policy, tokenizer=tokenizer, output_dir=best_model_dir)
                    runtime["release_memory"](train_device, eval_device)
            wandb.log(
                {
                    "eval_step": grpo_step,
                    "grpo_step": grpo_step,
                    **{f"eval/{key}": value for key, value in eval_summary.items() if isinstance(value, (int, float))},
                }
            )

        phase_timings["eval_sync_policy"] = eval_sync_seconds
        phase_timings["eval"] = eval_seconds
        phase_timings["step_total"] = time.perf_counter() - step_started_at
        step_summary["phase_seconds"] = {key: round(value, 6) for key, value in phase_timings.items()}
        step_summary["train_cuda_memory_mb"] = train_memory_snapshots
        runtime["write_json"](step_dir / "summary.json", step_summary)
        emit_step_trace(grpo_step, phase_timings, train_memory_snapshots)

    final_model_dir = output_dir / "final_model"
    if args.save_final_model:
        runtime["save_model_checkpoint"](policy=policy, tokenizer=tokenizer, output_dir=final_model_dir)
        runtime["release_memory"](train_device, eval_device)

    runtime["write_jsonl"](output_dir / "train_history.jsonl", train_history)
    runtime["write_jsonl"](output_dir / "eval_history.jsonl", eval_history)
    runtime["write_jsonl"](output_dir / "step_summaries.jsonl", step_summaries)
    for generation_artifact in generation_artifacts:
        runtime["write_jsonl"](
            output_dir / f"generation_logs_step_{generation_artifact['eval_step']:04d}.jsonl",
            generation_artifact["logs"],
        )

    summary = {
        "run_name": run_name,
        "model_path": str(model_path),
        "num_train_examples": len(train_examples),
        "num_eval_examples": len(eval_examples),
        "n_grpo_steps": args.n_grpo_steps,
        "rollout_batch_size": args.rollout_batch_size,
        "group_size": args.group_size,
        "epochs_per_rollout_batch": args.epochs_per_rollout_batch,
        "train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "loss_type": args.loss_type,
        "reward_function": args.reward_function,
        "length_normalization": args.length_normalization,
        "length_normalize_constant": length_normalize_constant,
        "use_std_normalization": args.use_std_normalization,
        "learning_rate": args.learning_rate,
        "wall_clock_seconds": time.perf_counter() - run_start_time,
        "final_accuracy": eval_history[-1]["accuracy"],
        "best_accuracy": max(record["accuracy"] for record in eval_history),
        "best_eval_step": None if best_eval_record is None else best_eval_record["eval_step"],
        "best_model_path": None if not args.save_final_model else str(best_model_dir),
        "final_model_path": None if not args.save_final_model else str(final_model_dir),
        "train_history_path": str(output_dir / "train_history.jsonl"),
        "eval_history_path": str(output_dir / "eval_history.jsonl"),
        "step_summaries_path": str(output_dir / "step_summaries.jsonl"),
    }
    runtime["write_json"](output_dir / "summary.json", summary)
    wandb_run.summary.update(summary)
    wandb_run.finish()
    return summary


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.n_grpo_steps = 1
        args.rollout_batch_size = 8
        args.group_size = 2
        args.train_batch_size = 8
        args.gradient_accumulation_steps = 4
        args.epochs_per_rollout_batch = 1
        args.max_eval_examples = 16
        args.num_log_generations = 4
        args.save_final_model = False
    summary = train_grpo(args)
    runtime = load_runtime_helpers()
    runtime["write_json"](
        args.output_root / make_run_name(args) / "sweep_summary.json",
        {
            "wandb_entity": args.wandb_entity,
            "wandb_project": args.wandb_project,
            "wandb_group": args.wandb_group,
            "runs": [summary],
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()