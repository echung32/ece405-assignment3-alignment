"""
Download the MATH (competition_math) dataset from HuggingFace and split into train/test.

Running:

```
python scripts/download_math.py [--output-dir data/math] [--test-size 2500] [--seed 42]
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


def main(output_dir: str, test_size: int, seed: int) -> None:
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

    test_size = min(test_size, len(examples))
    test_examples = examples[:test_size]
    train_examples = examples[test_size:]

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.jsonl")
    test_path = os.path.join(output_dir, "test.jsonl")

    with open(train_path, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")

    with open(test_path, "w") as f:
        for ex in test_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(train_examples)} train examples to {train_path}")
    print(f"Wrote {len(test_examples)} test examples  to {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and split the MATH dataset.")
    parser.add_argument(
        "--output-dir",
        default="data/math",
        help="Directory to write train.jsonl and test.jsonl (default: data/math)",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=2500,
        help="Number of examples to reserve for the test split (default: 2500, ~20%%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling before split (default: 42)",
    )
    args = parser.parse_args()
    main(args.output_dir, args.test_size, args.seed)
