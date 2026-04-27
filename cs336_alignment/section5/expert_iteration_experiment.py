from __future__ import annotations

import argparse
import gc
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import wandb
from torch.nn.utils import clip_grad_norm_
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.section4.get_response_log_probs import get_response_log_probs
from cs336_alignment.section4.log_generations import log_generations
from cs336_alignment.section4.sft_microbatch_train_step import sft_microbatch_train_step
from cs336_alignment.section4.tokenize_prompt_and_output import tokenize_prompt_and_output

DEFAULT_MODEL_CANDIDATES = [Path("data/Qwen/Qwen2.5-Math-1.5B")]
DEFAULT_TRAIN_PATH = Path("data/math/train.jsonl")
DEFAULT_VAL_PATH = Path("data/math/val.jsonl")
DEFAULT_PROMPT_PATH = Path("cs336_alignment/prompts/r1_zero.prompt")
DEFAULT_LOG_ROOT = Path("logs/section5")
DEFAULT_OUTPUT_ROOT = Path("data/section5/expert_iteration")
WANDB_ENTITY = "echung32-ece405"
WANDB_PROJECT = "ece405-alignment-expert-iteration"


@dataclass(frozen=True)
class RolloutExample:
    problem: str
    solution: str
    prompt: str
    response: str
    metrics: dict[str, float]
    rollout_index: int

    @property
    def is_correct(self) -> bool:
        return self.metrics["format_reward"] == 1.0 and self.metrics["answer_reward"] == 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Section 5 expert iteration on MATH.")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val-path", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--sft-epochs-per-step", type=int, default=1)
    parser.add_argument("--num-ei-steps", type=int, default=5)
    parser.add_argument("--rollouts-per-question", type=int, default=4)
    parser.add_argument("--questions-per-step", type=int, default=512)
    parser.add_argument("--num-evals-per-run", type=int, default=5)
    parser.add_argument("--max-eval-examples", type=int, default=1024)
    parser.add_argument("--num-log-generations", type=int, default=32)
    parser.add_argument("--generation-max-tokens", type=int, default=1024)
    parser.add_argument("--generation-temperature", type=float, default=1.0)
    parser.add_argument("--generation-top-p", type=float, default=1.0)
    parser.add_argument("--generation-min-tokens", type=int, default=4)
    parser.add_argument("--train-gpu-id", type=int, default=0)
    parser.add_argument("--eval-gpu-id", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--wandb-entity", type=str, default=WANDB_ENTITY)
    parser.add_argument("--wandb-project", type=str, default=WANDB_PROJECT)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-final-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def serialize_json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: serialize_json_value(nested_value) for key, nested_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_json_value(item) for item in value]
    return value


def load_runtime_helpers() -> dict[str, Any]:
    from cs336_alignment.math_baseline import evaluate_vllm, summarize_metrics
    from cs336_alignment.section4.sft_experiment import (
        build_microbatches,
        build_sft_examples,
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

    return {
        "build_microbatches": build_microbatches,
        "build_sft_examples": build_sft_examples,
        "evaluate_vllm": evaluate_vllm,
        "init_vllm": init_vllm,
        "load_jsonl": load_jsonl,
        "load_policy_into_vllm_instance": load_policy_into_vllm_instance,
        "load_policy_model": load_policy_model,
        "load_prompt_template": load_prompt_template,
        "load_tokenizer": load_tokenizer,
        "resolve_model_path": resolve_model_path,
        "save_model_checkpoint": save_model_checkpoint,
        "summarize_metrics": summarize_metrics,
        "write_json": write_json,
        "write_jsonl": write_jsonl,
    }


def build_eval_schedule(num_ei_steps: int, num_evals_per_run: int) -> set[int]:
    schedule = {0, num_ei_steps}
    for idx in range(1, num_evals_per_run + 1):
        schedule.add(max(1, math.ceil(num_ei_steps * idx / num_evals_per_run)))
    return schedule


def build_rollout_sampling_params(args: argparse.Namespace) -> Any:
    from vllm import SamplingParams

    return SamplingParams(
        temperature=args.generation_temperature,
        top_p=args.generation_top_p,
        max_tokens=args.generation_max_tokens,
        min_tokens=args.generation_min_tokens,
        n=args.rollouts_per_question,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=args.seed,
    )


def build_eval_sampling_params(args: argparse.Namespace) -> Any:
    from vllm import SamplingParams

    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.generation_max_tokens,
        min_tokens=args.generation_min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )


def release_memory(*devices: torch.device) -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return

    seen_device_indices: set[int] = set()
    for device in devices:
        if device.type != "cuda":
            continue
        device_index = torch.cuda.current_device() if device.index is None else device.index
        if device_index in seen_device_indices:
            continue
        with torch.cuda.device(device_index):
            torch.cuda.empty_cache()
        seen_device_indices.add(device_index)


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
    release_memory(*devices)
    return policy_version


def run_eval(
    policy: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    eval_examples: list[dict[str, Any]],
    sampling_params: Any,
    train_device: torch.device,
    llm: Any,
    num_log_generations: int,
    runtime: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prompts = [example["prompt"] for example in eval_examples]
    ground_truths = [example["solution"] for example in eval_examples]

    responses, metrics_list = runtime["evaluate_vllm"](
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
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
        reward_fn=r1_zero_reward_fn,
        device=train_device,
    )
    for log_entry, example in zip(generation_logs, eval_examples[:num_log_generations], strict=True):
        log_entry["problem"] = example["problem"]
        log_entry["level"] = example.get("level")
        log_entry["type"] = example.get("type")
    summary.update(generation_summary)
    release_memory(train_device)
    return summary, generation_logs


def sample_rollout_examples(
    train_examples: list[dict[str, Any]],
    questions_per_step: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    sample_size = min(questions_per_step, len(train_examples))
    if sample_size == len(train_examples):
        return list(train_examples)
    return rng.sample(train_examples, sample_size)


def generate_rollouts(
    llm: Any,
    prompts: list[str],
    sampling_params: Any,
) -> list[list[str]]:
    outputs = llm.generate(prompts, sampling_params)
    grouped_outputs: list[list[str]] = []
    for output in outputs:
        grouped_outputs.append([candidate.text for candidate in output.outputs])
    return grouped_outputs


def flatten_rollout_examples(
    sampled_examples: list[dict[str, Any]],
    grouped_responses: list[list[str]],
) -> list[RolloutExample]:
    flattened: list[RolloutExample] = []
    for example, responses in zip(sampled_examples, grouped_responses, strict=True):
        for rollout_index, response in enumerate(responses):
            metrics = r1_zero_reward_fn(response, example["solution"])
            flattened.append(
                RolloutExample(
                    problem=example["problem"],
                    solution=example["solution"],
                    prompt=example["prompt"],
                    response=response,
                    metrics=metrics,
                    rollout_index=rollout_index,
                )
            )
    return flattened


def summarize_rollouts(rollouts: list[RolloutExample]) -> dict[str, Any]:
    if not rollouts:
        return {
            "num_rollouts": 0,
            "num_correct_rollouts": 0,
            "num_unique_prompts_with_correct_rollout": 0,
            "mean_reward": 0.0,
            "mean_answer_reward": 0.0,
            "mean_format_reward": 0.0,
            "mean_response_length": 0.0,
        }

    unique_prompt_ids_with_correct = {
        (rollout.problem, rollout.solution)
        for rollout in rollouts
        if rollout.is_correct
    }
    response_lengths = [len(rollout.response) for rollout in rollouts]
    return {
        "num_rollouts": len(rollouts),
        "num_correct_rollouts": sum(1 for rollout in rollouts if rollout.is_correct),
        "num_unique_prompts_with_correct_rollout": len(unique_prompt_ids_with_correct),
        "mean_reward": sum(rollout.metrics["reward"] for rollout in rollouts) / len(rollouts),
        "mean_answer_reward": sum(rollout.metrics["answer_reward"] for rollout in rollouts) / len(rollouts),
        "mean_format_reward": sum(rollout.metrics["format_reward"] for rollout in rollouts) / len(rollouts),
        "mean_response_length": sum(response_lengths) / len(response_lengths),
    }


def serialize_rollouts(rollouts: list[RolloutExample]) -> list[dict[str, Any]]:
    return [
        {
            "problem": rollout.problem,
            "solution": rollout.solution,
            "prompt": rollout.prompt,
            "response": rollout.response,
            "rollout_index": rollout.rollout_index,
            "metrics": rollout.metrics,
        }
        for rollout in rollouts
    ]


def build_filtered_sft_examples(rollouts: list[RolloutExample]) -> list[dict[str, Any]]:
    return [
        {
            "problem": rollout.problem,
            "solution": rollout.solution,
            "prompt": rollout.prompt,
            "response": rollout.response,
            "source": "expert_iteration",
            "rollout_index": rollout.rollout_index,
            "metrics": rollout.metrics,
        }
        for rollout in rollouts
        if rollout.is_correct
    ]


def run_sft_epoch_chunk(
    policy: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    train_examples: list[dict[str, Any]],
    train_device: torch.device,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    rng: random.Random,
    train_step_offset: int,
    runtime: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    if not train_examples:
        return [], train_step_offset

    policy.train()
    microbatches = runtime["build_microbatches"](train_examples, per_device_batch_size, rng)
    train_history: list[dict[str, Any]] = []
    train_step = train_step_offset
    for chunk_start in range(0, len(microbatches), gradient_accumulation_steps):
        microbatch_chunk = microbatches[chunk_start : chunk_start + gradient_accumulation_steps]
        optimizer.zero_grad(set_to_none=True)
        chunk_losses: list[float] = []
        chunk_tokens = 0
        for microbatch in microbatch_chunk:
            prompts = [example["prompt"] for example in microbatch]
            responses = [example["response"] for example in microbatch]
            tokenized = tokenize_prompt_and_output(prompts, responses, tokenizer)
            input_ids = tokenized["input_ids"].to(train_device)
            labels = tokenized["labels"].to(train_device)
            response_mask = tokenized["response_mask"].to(train_device)
            scored = get_response_log_probs(
                model=policy,
                input_ids=input_ids,
                labels=labels,
                response_mask=response_mask,
                return_token_entropy=False,
            )
            loss, _ = sft_microbatch_train_step(
                policy_log_probs=scored["log_probs"],
                response_mask=response_mask,
                gradient_accumulation_steps=len(microbatch_chunk),
                normalize_constant=1.0,
            )
            chunk_losses.append(float(loss.detach().item()))
            chunk_tokens += int(response_mask.sum().item())
            del scored, loss, input_ids, labels, response_mask, tokenized

        release_memory(train_device)

        grad_norm = float(clip_grad_norm_(policy.parameters(), max_norm=1.0).item())
        optimizer.step()
        release_memory(train_device)
        train_step += 1
        train_history.append(
            {
                "train_step": train_step,
                "loss": sum(chunk_losses),
                "grad_norm": grad_norm,
                "response_tokens": chunk_tokens,
            }
        )
        del microbatch_chunk
    return train_history, train_step


def make_run_name(args: argparse.Namespace) -> str:
    return (
        f"g{args.rollouts_per_question}_"
        f"db{args.questions_per_step}_"
        f"ep{args.sft_epochs_per_step}_"
        f"steps{args.num_ei_steps}"
    )


def train_expert_iteration(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    runtime = load_runtime_helpers()

    model_path = runtime["resolve_model_path"](args.model_path)
    tokenizer = runtime["load_tokenizer"](model_path)
    prompt_template = runtime["load_prompt_template"](args.prompt_path)
    train_examples = runtime["build_sft_examples"](runtime["load_jsonl"](args.train_path), prompt_template)
    eval_examples = runtime["build_sft_examples"](runtime["load_jsonl"](args.val_path), prompt_template)
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
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    llm = runtime["init_vllm"](
        model_id=str(model_path),
        eval_gpu_id=args.eval_gpu_id,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    rollout_sampling_params = build_rollout_sampling_params(args)
    eval_sampling_params = build_eval_sampling_params(args)
    eval_schedule = build_eval_schedule(args.num_ei_steps, args.num_evals_per_run)

    if args.wandb_group is None:
        args.wandb_group = f"section5-ei-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

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
            "per_device_batch_size": args.per_device_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "sft_epochs_per_step": args.sft_epochs_per_step,
            "num_ei_steps": args.num_ei_steps,
            "rollouts_per_question": args.rollouts_per_question,
            "questions_per_step": args.questions_per_step,
        },
    )
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("ei_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")
    wandb.define_metric("ei/*", step_metric="ei_step")

    train_rng = random.Random(args.seed)
    train_history: list[dict[str, Any]] = []
    eval_history: list[dict[str, Any]] = []
    step_summaries: list[dict[str, Any]] = []
    generation_artifacts: list[dict[str, Any]] = []
    best_eval_record: dict[str, Any] | None = None
    best_model_dir = output_dir / "best_model"
    total_train_steps = 0
    synced_policy_version: int | None = None

    synced_policy_version = sync_policy_to_vllm_if_needed(
        policy,
        llm,
        total_train_steps,
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
        runtime=runtime,
    )
    initial_eval_record = {"eval_step": 0, "ei_step": 0, **initial_eval_summary}
    eval_history.append(initial_eval_record)
    generation_artifacts.append({"eval_step": 0, "logs": initial_generation_logs})
    best_eval_record = dict(initial_eval_record)
    if args.save_final_model:
        runtime["save_model_checkpoint"](policy=policy, tokenizer=tokenizer, output_dir=best_model_dir)
        release_memory(train_device, eval_device)
    wandb.log({
        "eval_step": 0,
        "ei_step": 0,
        **{f"eval/{key}": value for key, value in initial_eval_summary.items() if isinstance(value, (int, float))},
    })

    for ei_step in range(1, args.num_ei_steps + 1):
        sampled_examples = sample_rollout_examples(train_examples, args.questions_per_step, train_rng)
        prompts = [example["prompt"] for example in sampled_examples]
        synced_policy_version = sync_policy_to_vllm_if_needed(
            policy,
            llm,
            total_train_steps,
            synced_policy_version,
            runtime,
            train_device,
            eval_device,
        )
        grouped_responses = generate_rollouts(llm=llm, prompts=prompts, sampling_params=rollout_sampling_params)
        flattened_rollouts = flatten_rollout_examples(sampled_examples, grouped_responses)
        step_dir = output_dir / f"step_{ei_step:02d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        runtime["write_jsonl"](step_dir / "rollouts.jsonl", serialize_rollouts(flattened_rollouts))

        filtered_examples = build_filtered_sft_examples(flattened_rollouts)
        runtime["write_jsonl"](step_dir / "filtered_sft_examples.jsonl", filtered_examples)

        rollout_summary = summarize_rollouts(flattened_rollouts)
        rollout_summary["filtered_dataset_size"] = len(filtered_examples)
        rollout_summary["ei_step"] = ei_step
        del grouped_responses, prompts, sampled_examples
        release_memory(train_device, eval_device)

        step_train_history: list[dict[str, Any]] = []
        for epoch_idx in range(args.sft_epochs_per_step):
            epoch_history, total_train_steps = run_sft_epoch_chunk(
                policy=policy,
                tokenizer=tokenizer,
                optimizer=optimizer,
                train_examples=filtered_examples,
                train_device=train_device,
                per_device_batch_size=args.per_device_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                rng=train_rng,
                train_step_offset=total_train_steps,
                runtime=runtime,
            )
            for record in epoch_history:
                record["ei_step"] = ei_step
                record["sft_epoch_within_ei"] = epoch_idx + 1
            step_train_history.extend(epoch_history)
            train_history.extend(epoch_history)
            for record in epoch_history:
                wandb.log(
                    {
                        "train_step": record["train_step"],
                        "ei_step": ei_step,
                        "train/loss": record["loss"],
                        "train/grad_norm": record["grad_norm"],
                        "train/response_tokens": record["response_tokens"],
                    }
                )

                synced_policy_version = None if step_train_history else synced_policy_version
                release_memory(train_device, eval_device)

        runtime["write_jsonl"](step_dir / "train_history.jsonl", step_train_history)
        runtime["write_json"](step_dir / "summary.json", rollout_summary)
        step_summaries.append(rollout_summary)
        wandb.log(
            {
                "ei_step": ei_step,
                **{f"ei/{key}": value for key, value in rollout_summary.items() if isinstance(value, (int, float))},
            }
        )

        if ei_step in eval_schedule:
            synced_policy_version = sync_policy_to_vllm_if_needed(
                policy,
                llm,
                total_train_steps,
                synced_policy_version,
                runtime,
                train_device,
                eval_device,
            )
            eval_summary, generation_logs = run_eval(
                policy=policy,
                tokenizer=tokenizer,
                eval_examples=eval_examples,
                sampling_params=eval_sampling_params,
                train_device=train_device,
                llm=llm,
                num_log_generations=args.num_log_generations,
                runtime=runtime,
            )
            eval_record = {"eval_step": ei_step, "ei_step": ei_step, **eval_summary}
            eval_history.append(eval_record)
            generation_artifacts.append({"eval_step": ei_step, "logs": generation_logs})
            if best_eval_record is None or eval_record["accuracy"] > best_eval_record["accuracy"]:
                best_eval_record = dict(eval_record)
                if args.save_final_model:
                    runtime["save_model_checkpoint"](policy=policy, tokenizer=tokenizer, output_dir=best_model_dir)
                    release_memory(train_device, eval_device)
            wandb.log(
                {
                    "eval_step": ei_step,
                    "ei_step": ei_step,
                    **{f"eval/{key}": value for key, value in eval_summary.items() if isinstance(value, (int, float))},
                }
            )

    final_model_dir = output_dir / "final_model"
    if args.save_final_model:
        runtime["save_model_checkpoint"](policy=policy, tokenizer=tokenizer, output_dir=final_model_dir)
        release_memory(train_device, eval_device)

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
        "num_ei_steps": args.num_ei_steps,
        "rollouts_per_question": args.rollouts_per_question,
        "questions_per_step": args.questions_per_step,
        "sft_epochs_per_step": args.sft_epochs_per_step,
        "learning_rate": args.learning_rate,
        "per_device_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "final_accuracy": eval_history[-1]["accuracy"],
        "best_accuracy": max(record["accuracy"] for record in eval_history),
        "best_eval_step": None if best_eval_record is None else best_eval_record["eval_step"],
        "final_mean_token_entropy": eval_history[-1].get("mean_token_entropy"),
        "best_mean_token_entropy": None if best_eval_record is None else best_eval_record.get("mean_token_entropy"),
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
        args.num_ei_steps = 1
        args.questions_per_step = 16
        args.rollouts_per_question = 2
        args.sft_epochs_per_step = 1
        args.max_eval_examples = 16
        args.num_log_generations = 4
        args.save_final_model = False
    summary = train_expert_iteration(args)
    sweep_summary = {
        "wandb_entity": args.wandb_entity,
        "wandb_project": args.wandb_project,
        "wandb_group": args.wandb_group,
        "runs": [summary],
    }
    runtime = load_runtime_helpers()
    runtime["write_json"](args.output_root / make_run_name(args) / "sweep_summary.json", sweep_summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()