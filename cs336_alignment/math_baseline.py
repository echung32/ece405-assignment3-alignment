"""
uv run scripts/math_baseline.py --model-path data/Qwen/Qwen2.5-Math-1.5B
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Callable


from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import extract_answer, r1_zero_reward_fn

DEFAULT_MODEL_CANDIDATES = [
    Path("data/Qwen/Qwen2.5-Math-1.5B"),
]
DEFAULT_DATASET_PATH = Path("data/math/test.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/math/math_baseline_predictions.jsonl")
DEFAULT_SUMMARY_PATH = Path("data/math/math_baseline_summary.json")
PROMPT_PATH = Path("cs336_alignment/prompts/r1_zero.prompt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a zero-shot MATH baseline with local vLLM inference.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=(
            "Local HuggingFace model path to evaluate. If omitted, the script picks "
            "the first existing default candidate under data/."
        ),
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the MATH JSONL split to evaluate.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write per-example generations and metrics as JSONL.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Path to write aggregate metrics as JSON.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--min-tokens", type=int, default=4)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="Fraction of GPU memory vLLM is allowed to reserve.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap for a quick smoke test over the dataset.",
    )
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

    searched_paths = "\n".join(f"  - {candidate}" for candidate in DEFAULT_MODEL_CANDIDATES)
    raise FileNotFoundError(
        "No default model directory was found. Pass --model-path or place a model in one of:\n"
        f"{searched_paths}"
    )


def load_examples(input_path: Path, max_examples: int | None) -> list[dict]:
    examples = []
    with input_path.open() as infile:
        for line in infile:
            examples.append(json.loads(line))
            if max_examples is not None and len(examples) >= max_examples:
                break
    return examples


def load_prompt_template(prompt_path: Path) -> str:
    return prompt_path.read_text().strip()


def build_prompts(examples: list[dict], prompt_template: str) -> list[str]:
    return [prompt_template.format(question=example["problem"]) for example in examples]


def categorize_result(metrics: dict[str, float]) -> str:
    if metrics["format_reward"] == 1.0 and metrics["answer_reward"] == 1.0:
        return "correct_with_format"
    if metrics["format_reward"] == 1.0 and metrics["answer_reward"] == 0.0:
        return "formatted_but_incorrect"
    return "unformatted"


def extract_model_answer_text(response: str) -> str | None:
    if "<answer>" not in response or "</answer>" not in response:
        return None
    return response.split("<answer>")[-1].replace("</answer>", "").strip()


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    ground_truths: list[str],
    eval_sampling_params: SamplingParams,
) -> tuple[list[str], list[dict[str, float]]]:
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    metrics = [reward_fn(response, ground_truth) for response, ground_truth in zip(responses, ground_truths)]
    return responses, metrics


def summarize_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float | int | dict[str, int]]:
    category_counts = Counter(categorize_result(metrics) for metrics in metrics_list)
    return {
        "num_examples": len(metrics_list),
        "accuracy": mean(metrics["answer_reward"] for metrics in metrics_list),
        "mean_reward": mean(metrics["reward"] for metrics in metrics_list),
        "mean_format_reward": mean(metrics["format_reward"] for metrics in metrics_list),
        "category_counts": dict(category_counts),
    }


def write_outputs(
    output_path: Path,
    examples: list[dict],
    prompts: list[str],
    responses: list[str],
    metrics_list: list[dict[str, float]],
    model_path: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as outfile:
        for example, prompt, response, metrics in zip(examples, prompts, responses, metrics_list):
            outfile.write(
                json.dumps(
                    {
                        **example,
                        "model_path": model_path,
                        "prompt": prompt,
                        "response": response,
                        "answer_text": extract_model_answer_text(response),
                        "parsed_answer": extract_answer(response),
                        "metrics": metrics,
                        "category": categorize_result(metrics),
                    }
                )
                + "\n"
            )


def write_summary(summary_path: Path, summary: dict) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as outfile:
        json.dump(summary, outfile, indent=2, sort_keys=True)
        outfile.write("\n")


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model_path)

    if not args.input_path.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {args.input_path}")

    examples = load_examples(args.input_path, args.max_examples)
    prompt_template = load_prompt_template(PROMPT_PATH)
    prompts = build_prompts(examples, prompt_template)
    ground_truths = [example["solution"] for example in examples]

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    vllm_model = LLM(
        model=str(model_path),
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )

    responses, metrics_list = evaluate_vllm(
        vllm_model=vllm_model,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampling_params,
    )

    summary = summarize_metrics(metrics_list)
    summary.update(
        {
            "model_path": str(model_path),
            "input_path": str(args.input_path),
            "output_path": str(args.output_path),
            "summary_path": str(args.summary_path),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "min_tokens": args.min_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        }
    )

    write_outputs(
        output_path=args.output_path,
        examples=examples,
        prompts=prompts,
        responses=responses,
        metrics_list=metrics_list,
        model_path=str(model_path),
    )
    write_summary(args.summary_path, summary)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()