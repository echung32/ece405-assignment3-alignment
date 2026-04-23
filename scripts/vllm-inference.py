"""
uv run python scripts/vllm-inference.py
uv run python scripts/vllm-inference.py --model-path data/Qwen/Qwen2.5-0.5B
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vllm import LLM, SamplingParams

DEFAULT_MODEL_CANDIDATES = [
    Path("data/Qwen/Qwen2.5-0.5B"),
    Path("data/Qwen/Qwen2.5-3B-Instruct"),
]

DEFAULT_PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the assignment Section 3.1 vLLM offline inference smoke test.",
    )
    parser.add_argument(
        "--model-path",
        dest="model_paths",
        action="append",
        help=(
            "Path to a local HuggingFace model directory. Repeat this flag to test "
            "multiple models. Defaults to the expected Qwen directories under data/."
        ),
    )
    parser.add_argument(
        "--prompt",
        dest="prompts",
        action="append",
        help="Prompt to generate from. Repeat this flag to override the default prompts.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="Fraction of GPU memory vLLM is allowed to reserve.",
    )
    parser.add_argument(
        "--stop",
        action="append",
        default=["\n"],
        help="Stop string to pass to vLLM. Repeat to provide multiple stop strings.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    return parser.parse_args()


def ensure_models_exist(model_paths: list[Path]) -> None:
    missing_paths = [path for path in model_paths if not path.exists()]
    if missing_paths:
        missing = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(
            "Missing local model directories. Download the models into data/ first:\n"
            f"{missing}"
        )


def default_existing_models() -> list[Path]:
    model_paths = [candidate for candidate in DEFAULT_MODEL_CANDIDATES if candidate.exists()]
    if model_paths:
        return model_paths

    searched = "\n".join(f"  - {candidate}" for candidate in DEFAULT_MODEL_CANDIDATES)
    raise FileNotFoundError(
        "No default local Qwen model directories were found. Download the models and place them in one of:\n"
        f"{searched}"
    )


def run_smoke_test(
    model_paths: list[Path],
    prompts: list[str],
    temperature: float,
    top_p: float,
    max_tokens: int,
    stop: list[str],
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
) -> None:
    ensure_models_exist(model_paths)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=stop,
    )

    for model_path in model_paths:
        print(f"\n=== Testing model: {model_path} ===")
        llm = LLM(
            model=str(model_path),
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )
        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            generated_text = output.outputs[0].text
            print(f"Prompt: {output.prompt!r}")
            print(f"Generated text: {generated_text!r}")
            print()


def main() -> None:
    args = parse_args()
    model_paths = [Path(path) for path in args.model_paths] if args.model_paths else default_existing_models()
    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS
    run_smoke_test(
        model_paths=model_paths,
        prompts=prompts,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=args.stop,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )


if __name__ == "__main__":
    main()