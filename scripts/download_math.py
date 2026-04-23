"""
Download the MATH (competition_math) dataset from HuggingFace and split into train/val/test.

Running:

```
python scripts/download_math.py [--output-dir data/math] [--val-size 1250] [--test-size 1250] [--seed 42]
```

Dataset: https://huggingface.co/datasets/qwedsacf/competition_math
Fields: problem, solution, level, type
"""
import argparse
import json
import os
import random

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


def write_jsonl(path: str, examples: list[dict]) -> None:
    with open(path, "w") as outfile:
        for example in examples:
            outfile.write(json.dumps(example) + "\n")


def main(output_dir: str, val_size: int, test_size: int, seed: int) -> None:
    print(f"Downloading qwedsacf/competition_math from HuggingFace...")
    parquet_path = hf_hub_download(
        repo_id="qwedsacf/competition_math",
        repo_type="dataset",
        revision="refs/convert/parquet",
        filename="default/train/0000.parquet",
    )
    table = pq.read_table(parquet_path)
    examples = table.to_pylist()
    print(f"Downloaded {len(examples)} examples.")

    rng = random.Random(seed)
    rng.shuffle(examples)

    if val_size < 0 or test_size < 0:
        raise ValueError("val_size and test_size must be non-negative")

    held_out_size = min(val_size + test_size, len(examples))
    val_size = min(val_size, held_out_size)
    test_size = held_out_size - val_size

    val_examples = examples[:val_size]
    test_examples = examples[val_size : val_size + test_size]
    train_examples = examples[val_size + test_size :]

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")
    test_path = os.path.join(output_dir, "test.jsonl")

    write_jsonl(train_path, train_examples)
    write_jsonl(val_path, val_examples)
    write_jsonl(test_path, test_examples)

    print(f"Wrote {len(train_examples)} train examples to {train_path}")
    print(f"Wrote {len(val_examples)} val examples    to {val_path}")
    print(f"Wrote {len(test_examples)} test examples  to {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and split the MATH dataset.")
    parser.add_argument(
        "--output-dir",
        default="data/math",
        help="Directory to write train.jsonl, val.jsonl, and test.jsonl (default: data/math)",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=1250,
        help="Number of examples to reserve for the validation split (default: 1250, ~10%%)",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=1250,
        help="Number of examples to reserve for the test split (default: 1250, ~10%%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling before split (default: 42)",
    )
    args = parser.parse_args()
    main(args.output_dir, args.val_size, args.test_size, args.seed)
