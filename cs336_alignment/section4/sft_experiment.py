from __future__ import annotations

import argparse
from functools import partial
import json
import math
import os
import random
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import wandb
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams
from vllm.utils.torch_utils import set_random_seed as vllm_set_random_seed

from cs336_alignment.drgrpo_grader import extract_answer, r1_zero_reward_fn
from cs336_alignment.math_baseline import evaluate_vllm, summarize_metrics
from cs336_alignment.section4.get_response_log_probs import get_response_log_probs
from cs336_alignment.section4.log_generations import log_generations
from cs336_alignment.section4.sft_microbatch_train_step import sft_microbatch_train_step
from cs336_alignment.section4.tokenize_prompt_and_output import tokenize_prompt_and_output

DEFAULT_MODEL_CANDIDATES = [Path("data/Qwen/Qwen2.5-Math-1.5B")]
DEFAULT_TRAIN_PATH = Path("data/math/train.jsonl")
DEFAULT_VAL_PATH = Path("data/math/val.jsonl")
DEFAULT_PROMPT_PATH = Path("cs336_alignment/prompts/r1_zero.prompt")
DEFAULT_OUTPUT_ROOT = Path("data/section4/sft_experiment")
DEFAULT_LOG_ROOT = Path("logs/section4")
DEFAULT_SUBSET_SIZES = ["128", "256", "512", "1024", "full"]
WANDB_ENTITY = "echung32-ece405"
WANDB_PROJECT = "ece405-alignment-sft"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Section 4 supervised finetuning on MATH.")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val-path", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--subset-sizes", nargs="+", default=DEFAULT_SUBSET_SIZES)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--num-evals-per-run", type=int, default=5)
    parser.add_argument("--max-eval-examples", type=int, default=None)
    parser.add_argument("--num-log-generations", type=int, default=32)
    parser.add_argument("--generation-max-tokens", type=int, default=1024)
    parser.add_argument("--generation-temperature", type=float, default=0.0)
    parser.add_argument("--generation-top-p", type=float, default=1.0)
    parser.add_argument("--train-gpu-id", type=int, default=0)
    parser.add_argument("--eval-gpu-id", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--wandb-entity", type=str, default=WANDB_ENTITY)
    parser.add_argument("--wandb-project", type=str, default=WANDB_PROJECT)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--skip-filtered-run", action="store_true")
    parser.add_argument("--only-filtered-run", action="store_true")
    parser.add_argument("--save-final-model", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_model_path(model_path_arg: str | None) -> Path:
    if model_path_arg is not None:
        model_path = Path(model_path_arg)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        return model_path
    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No local model directory was found. Expected one of: "
        + ", ".join(str(candidate) for candidate in DEFAULT_MODEL_CANDIDATES)
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as infile:
        return [json.loads(line) for line in infile]


def load_prompt_template(path: Path) -> str:
    return path.read_text().strip()


def prompt_opens_reasoning_tag(prompt_template: str) -> bool:
    return prompt_template.rstrip().endswith("<think>")


def build_supervised_response(solution: str, prompt_template: str) -> str:
    reasoning = solution.strip()
    answer_text = extract_answer(reasoning)
    if answer_text is None:
        answer_text = reasoning.splitlines()[-1].strip()
    if prompt_opens_reasoning_tag(prompt_template):
        return f"{reasoning}</think> <answer> {answer_text} </answer>"
    return f"<think>{reasoning}</think> <answer> {answer_text} </answer>"


def build_sft_examples(examples: list[dict[str, Any]], prompt_template: str) -> list[dict[str, Any]]:
    output_examples: list[dict[str, Any]] = []
    for example in examples:
        output_examples.append(
            {
                **example,
                "prompt": prompt_template.format(question=example["problem"]),
                "response": build_supervised_response(example["solution"], prompt_template=prompt_template),
            }
        )
    return output_examples


def filter_correct_examples(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered = []
    for example in examples:
        reward = r1_zero_reward_fn(example["response"], example["solution"])
        if reward["format_reward"] == 1.0 and reward["answer_reward"] == 1.0:
            filtered.append(example)
    return filtered


def parse_subset_sizes(values: list[str], total_examples: int) -> list[int]:
    parsed: list[int] = []
    for value in values:
        if value == "full":
            subset_size = total_examples
        else:
            subset_size = min(int(value), total_examples)
        if subset_size not in parsed:
            parsed.append(subset_size)
    return parsed


def infer_training_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def load_tokenizer(model_path: Path) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_policy_model(model_path: Path, train_device: torch.device) -> PreTrainedModel:
    dtype = infer_training_dtype()
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": dtype,
    }
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.to(train_device)
    return model


def build_microbatches(
    examples: list[dict[str, Any]],
    batch_size: int,
    rng: random.Random,
) -> list[list[dict[str, Any]]]:
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    return [
        [examples[index] for index in indices[start : start + batch_size]]
        for start in range(0, len(indices), batch_size)
    ]


def build_eval_schedule(total_optimizer_steps: int, num_evals_per_run: int) -> set[int]:
    schedule = {total_optimizer_steps}
    for idx in range(1, num_evals_per_run + 1):
        schedule.add(max(1, math.ceil(total_optimizer_steps * idx / num_evals_per_run)))
    return schedule


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as outfile:
        json.dump(payload, outfile, indent=2, sort_keys=True)
        outfile.write("\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as outfile:
        for record in records:
            outfile.write(json.dumps(record) + "\n")


def save_model_checkpoint(
    policy: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


@contextmanager
def temporary_env(overrides: dict[str, str]) -> Any:
    original_values = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, original_value in original_values.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


def init_vllm(model_id: str, eval_gpu_id: int, seed: int, gpu_memory_utilization: float, tensor_parallel_size: int) -> LLM:
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    vllm_set_random_seed(seed)
    with temporary_env({"CUDA_VISIBLE_DEVICES": str(eval_gpu_id)}):
        return LLM(
            model=model_id,
            dtype=torch.bfloat16 if infer_training_dtype() == torch.bfloat16 else torch.float32,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
        )


def load_weights_into_vllm_model(
    model: torch.nn.Module,
    state_dict_items: tuple[tuple[str, torch.Tensor], ...],
) -> None:
    model.load_weights(state_dict_items)


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM) -> None:
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    state_dict_items = tuple((name, tensor.detach().cpu()) for name, tensor in policy.state_dict().items())
    llm.apply_model(partial(load_weights_into_vllm_model, state_dict_items=state_dict_items))


def run_eval(
    policy: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    eval_examples: list[dict[str, Any]],
    sampling_params: SamplingParams,
    train_device: torch.device,
    llm: LLM,
    num_log_generations: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prompts = [example["prompt"] for example in eval_examples]
    ground_truths = [example["solution"] for example in eval_examples]

    load_policy_into_vllm_instance(policy, llm)
    responses, metrics_list = evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampling_params,
    )

    summary = summarize_metrics(metrics_list)
    logged_examples = eval_examples[:num_log_generations]
    logged_prompts = prompts[:num_log_generations]
    logged_responses = responses[:num_log_generations]
    logged_ground_truths = ground_truths[:num_log_generations]
    generation_logs, generation_summary = log_generations(
        model=policy,
        tokenizer=tokenizer,
        prompts=logged_prompts,
        responses=logged_responses,
        ground_truths=logged_ground_truths,
        reward_fn=r1_zero_reward_fn,
        device=train_device,
    )
    for log_entry, example in zip(generation_logs, logged_examples, strict=True):
        log_entry["problem"] = example["problem"]
        log_entry["level"] = example.get("level")
        log_entry["type"] = example.get("type")
    summary.update(generation_summary)
    summary["eval_backend"] = "vllm"
    return summary, generation_logs


def train_single_run(
    args: argparse.Namespace,
    model_path: Path,
    tokenizer: PreTrainedTokenizerBase,
    train_examples: list[dict[str, Any]],
    eval_examples: list[dict[str, Any]],
    run_name: str,
    output_dir: Path,
) -> dict[str, Any]:
    train_device = torch.device(f"cuda:{args.train_gpu_id}" if torch.cuda.is_available() else "cpu")
    policy = load_policy_model(model_path=model_path, train_device=train_device)
    policy.train()

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    sampling_params = SamplingParams(
        temperature=args.generation_temperature,
        top_p=args.generation_top_p,
        max_tokens=args.generation_max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    llm = init_vllm(
        model_id=str(model_path),
        eval_gpu_id=args.eval_gpu_id,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    microbatches_per_epoch = math.ceil(len(train_examples) / args.per_device_batch_size)
    total_optimizer_steps = math.ceil(microbatches_per_epoch / args.gradient_accumulation_steps) * args.num_epochs
    eval_schedule = build_eval_schedule(total_optimizer_steps, args.num_evals_per_run)

    config = {
        "run_name": run_name,
        "model_path": str(model_path),
        "num_train_examples": len(train_examples),
        "num_eval_examples": len(eval_examples),
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "per_device_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_epochs": args.num_epochs,
        "num_evals_per_run": args.num_evals_per_run,
        "eval_backend_used": "vllm",
        "wandb_entity": args.wandb_entity,
        "wandb_project": args.wandb_project,
    }
    wandb_run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=args.wandb_group,
        name=run_name,
        config=config,
    )
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    train_history: list[dict[str, Any]] = []
    eval_history: list[dict[str, Any]] = []
    generation_artifacts: list[dict[str, Any]] = []
    best_eval_record: dict[str, Any] | None = None
    best_model_dir = output_dir / "best_model"
    optimizer_step = 0
    train_rng = random.Random(args.seed)

    initial_eval_summary, initial_generation_logs = run_eval(
        policy=policy,
        tokenizer=tokenizer,
        eval_examples=eval_examples,
        sampling_params=sampling_params,
        train_device=train_device,
        llm=llm,
        num_log_generations=args.num_log_generations,
    )
    initial_eval_record = {"eval_step": 0, **initial_eval_summary}
    eval_history.append(initial_eval_record)
    generation_artifacts.append({"eval_step": 0, "logs": initial_generation_logs})
    best_eval_record = dict(initial_eval_record)
    if args.save_final_model:
        save_model_checkpoint(policy=policy, tokenizer=tokenizer, output_dir=best_model_dir)
    wandb.log({"eval_step": 0, **{f"eval/{key}": value for key, value in initial_eval_summary.items() if isinstance(value, (int, float))}})

    for epoch_idx in range(args.num_epochs):
        epoch_microbatches = build_microbatches(
            examples=train_examples,
            batch_size=args.per_device_batch_size,
            rng=train_rng,
        )
        for chunk_start in range(0, len(epoch_microbatches), args.gradient_accumulation_steps):
            microbatch_chunk = epoch_microbatches[chunk_start : chunk_start + args.gradient_accumulation_steps]
            optimizer.zero_grad(set_to_none=True)
            chunk_losses: list[float] = []
            chunk_response_tokens = 0
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
                    return_token_entropy=False,
                )
                loss, _ = sft_microbatch_train_step(
                    policy_log_probs=scored["log_probs"],
                    response_mask=response_mask,
                    gradient_accumulation_steps=len(microbatch_chunk),
                    normalize_constant=1.0,
                )
                chunk_losses.append(float(loss.detach().item()))
                chunk_response_tokens += int(response_mask.sum().item())

            grad_norm = float(clip_grad_norm_(policy.parameters(), max_norm=1.0).item())
            optimizer.step()
            optimizer_step += 1
            train_record = {
                "train_step": optimizer_step,
                "epoch": epoch_idx + 1,
                "loss": sum(chunk_losses),
                "grad_norm": grad_norm,
                "response_tokens": chunk_response_tokens,
            }
            train_history.append(train_record)
            wandb.log(
                {
                    "train_step": optimizer_step,
                    "train/loss": train_record["loss"],
                    "train/grad_norm": grad_norm,
                    "train/response_tokens": chunk_response_tokens,
                }
            )

            if optimizer_step in eval_schedule:
                eval_summary, generation_logs = run_eval(
                    policy=policy,
                    tokenizer=tokenizer,
                    eval_examples=eval_examples,
                    sampling_params=sampling_params,
                    train_device=train_device,
                    llm=llm,
                    num_log_generations=args.num_log_generations,
                )
                eval_record = {"eval_step": optimizer_step, **eval_summary}
                eval_history.append(eval_record)
                generation_artifacts.append({"eval_step": optimizer_step, "logs": generation_logs})
                if best_eval_record is None or eval_record["accuracy"] > best_eval_record["accuracy"]:
                    best_eval_record = dict(eval_record)
                    if args.save_final_model:
                        save_model_checkpoint(policy=policy, tokenizer=tokenizer, output_dir=best_model_dir)
                wandb.log(
                    {
                        "eval_step": optimizer_step,
                        **{f"eval/{key}": value for key, value in eval_summary.items() if isinstance(value, (int, float))},
                    }
                )

    final_model_dir = output_dir / "final_model"
    if args.save_final_model:
        save_model_checkpoint(policy=policy, tokenizer=tokenizer, output_dir=final_model_dir)

    write_jsonl(output_dir / "train_history.jsonl", train_history)
    write_jsonl(output_dir / "eval_history.jsonl", eval_history)
    for generation_artifact in generation_artifacts:
        write_jsonl(
            output_dir / f"generation_logs_step_{generation_artifact['eval_step']:04d}.jsonl",
            generation_artifact["logs"],
        )

    summary = {
        **config,
        "final_accuracy": eval_history[-1]["accuracy"],
        "best_accuracy": max(record["accuracy"] for record in eval_history),
        "best_eval_step": None if best_eval_record is None else best_eval_record["eval_step"],
        "best_model_path": None if not args.save_final_model else str(best_model_dir),
        "final_model_path": None if not args.save_final_model else str(final_model_dir),
        "total_optimizer_steps": total_optimizer_steps,
        "train_history_path": str(output_dir / "train_history.jsonl"),
        "eval_history_path": str(output_dir / "eval_history.jsonl"),
    }
    write_json(output_dir / "summary.json", summary)
    wandb_run.summary.update(summary)
    wandb_run.finish()
    return {
        "summary": summary,
        "train_history": train_history,
        "eval_history": eval_history,
    }


def main() -> None:
    args = parse_args()
    if args.only_filtered_run:
        args.skip_filtered_run = False
    if args.wandb_group is None:
        args.wandb_group = f"section4-sft-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if args.smoke_test:
        args.subset_sizes = ["128"]
        args.max_eval_examples = 32
        args.num_evals_per_run = 2
        args.num_log_generations = 8
        args.num_epochs = 1
        args.skip_filtered_run = True
        args.save_final_model = False

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_path = resolve_model_path(args.model_path)
    tokenizer = load_tokenizer(model_path)
    prompt_template = load_prompt_template(args.prompt_path)
    train_examples = build_sft_examples(load_jsonl(args.train_path), prompt_template)
    eval_examples = build_sft_examples(load_jsonl(args.val_path), prompt_template)
    if args.max_eval_examples is not None:
        eval_examples = eval_examples[: args.max_eval_examples]

    shuffled_train_examples = list(train_examples)
    random.Random(args.seed).shuffle(shuffled_train_examples)
    subset_sizes = parse_subset_sizes(args.subset_sizes, total_examples=len(shuffled_train_examples))

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)

    standard_runs: list[dict[str, Any]] = []
    if not args.only_filtered_run:
        for subset_size in subset_sizes:
            run_label = "full" if subset_size == len(shuffled_train_examples) else str(subset_size)
            run_name = f"unfiltered_{run_label}"
            run_output_dir = args.output_root / run_name
            subset_examples = shuffled_train_examples[:subset_size]
            write_jsonl(run_output_dir / "train_examples.jsonl", subset_examples)
            run_result = train_single_run(
                args=args,
                model_path=model_path,
                tokenizer=tokenizer,
                train_examples=subset_examples,
                eval_examples=eval_examples,
                run_name=run_name,
                output_dir=run_output_dir,
            )
            standard_runs.append(run_result)

    filtered_run = None
    if not args.skip_filtered_run:
        filtered_examples = filter_correct_examples(shuffled_train_examples)
        filtered_run_name = "filtered_full"
        filtered_output_dir = args.output_root / filtered_run_name
        write_jsonl(filtered_output_dir / "train_examples.jsonl", filtered_examples)
        filtered_run = train_single_run(
            args=args,
            model_path=model_path,
            tokenizer=tokenizer,
            train_examples=filtered_examples,
            eval_examples=eval_examples,
            run_name=filtered_run_name,
            output_dir=filtered_output_dir,
        )

    sweep_summary = {
        "model_path": str(model_path),
        "wandb_entity": args.wandb_entity,
        "wandb_project": args.wandb_project,
        "wandb_group": args.wandb_group,
        "standard_runs": [run["summary"] for run in standard_runs],
        "filtered_run": None if filtered_run is None else filtered_run["summary"],
        "filtered_dataset_size": None if filtered_run is None else filtered_run["summary"]["num_train_examples"],
    }
    write_json(args.output_root / "sweep_summary.json", sweep_summary)
    print(json.dumps(sweep_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
