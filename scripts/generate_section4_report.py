from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/section4/sft_experiment"
OUTPUT_DIR = ROOT / "data/section4/analysis"

SERIAL_RUN_DIR = DATA_ROOT / "section4_sft_all_20260423-0010_norm1_serial"
HPARAM_RUN_DIR = DATA_ROOT / "section4_sft_all_20260423-0010_norm1_hparam"
FILTERED_RUN_DIR = (
    DATA_ROOT / "section4_sft_filtered_full_20260423-2em5" / "filtered_full"
)


@dataclass
class RunSummary:
    label: str
    summary_path: Path
    summary: dict
    eval_history: list[dict]

    @property
    def best_accuracy(self) -> float:
        return float(self.summary["best_accuracy"])

    @property
    def final_accuracy(self) -> float:
        return float(self.summary["final_accuracy"])

    @property
    def best_eval_step(self) -> int:
        return int(self.summary["best_eval_step"])

    @property
    def steps_per_epoch(self) -> int:
        per_device = int(self.summary["per_device_batch_size"])
        grad_accum = int(self.summary["gradient_accumulation_steps"])
        num_train = int(self.summary["num_train_examples"])
        return math.ceil(math.ceil(num_train / per_device) / grad_accum)

    @property
    def best_epoch_fraction(self) -> float:
        return self.best_eval_step / self.steps_per_epoch


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def resolve_artifact_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return ROOT / path


def load_run_summary(summary_path: Path, label: str) -> RunSummary:
    summary = load_json(summary_path)
    eval_history = load_jsonl(resolve_artifact_path(summary["eval_history_path"]))
    return RunSummary(
        label=label,
        summary_path=summary_path,
        summary=summary,
        eval_history=eval_history,
    )


def maybe_load_run_summary(summary_path: Path, label: str) -> RunSummary | None:
    if not summary_path.exists():
        return None
    return load_run_summary(summary_path, label)


def build_series(
    runs: list[RunSummary],
    label_fn,
    x_fn=None,
) -> list[tuple[str, list[tuple[float, float]], tuple[float, float, float, float]]]:
    cmap = plt.get_cmap("tab10")
    series = []
    if x_fn is None:
        x_fn = lambda run, item: float(item["eval_step"])
    for index, run in enumerate(runs):
        points = [
            (x_fn(run, item), float(item["accuracy"]))
            for item in run.eval_history
            if item.get("accuracy") is not None
        ]
        series.append((label_fn(run), points, cmap(index % 10)))
    return series


def render_line_chart(
    title: str,
    x_label: str,
    y_label: str,
    series: list[
        tuple[str, list[tuple[float, float]], tuple[float, float, float, float]]
    ],
    output_path: Path,
) -> None:
    figure, ax = plt.subplots(figsize=(10.5, 6.2), dpi=180)
    for label, points, color in series:
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        ax.plot(
            x_values,
            y_values,
            label=label,
            linewidth=2.2,
            marker="o",
            markersize=3.2,
            color=color,
        )

    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.margins(x=0.02)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def build_commentary(
    dataset_runs: list[RunSummary],
    hparam_runs: list[RunSummary],
    filtered_run: RunSummary,
    best_hparam_run: RunSummary,
) -> str:
    best_dataset_run = max(dataset_runs, key=lambda run: run.best_accuracy)
    best_full_unfiltered = next(run for run in hparam_runs if run.label == "2e-5")
    filtered_delta = filtered_run.best_accuracy - best_full_unfiltered.best_accuracy
    lines = [
        f"- The strongest unfiltered SFT run was `{best_hparam_run.label}` with best validation accuracy {best_hparam_run.best_accuracy:.4f} at step {best_hparam_run.best_eval_step} (epoch {best_hparam_run.best_epoch_fraction:.2f}).",
        f"- Within the dataset-size sweep, the best run was `{best_dataset_run.label}` with best validation accuracy {best_dataset_run.best_accuracy:.4f}, so the full dataset still performed best among the fixed-learning-rate serial runs.",
        f"- The filtered-full run kept {filtered_run.summary['num_train_examples']} examples and reached {filtered_run.best_accuracy:.4f} best validation accuracy, which is {filtered_delta:.4f} relative to the unfiltered full-data `2e-5` run at {best_full_unfiltered.best_accuracy:.4f}.",
    ]
    if filtered_delta >= 0:
        lines.append(
            "- On these artifacts, filtering did not hurt peak validation accuracy and may be worth keeping if the later RL stages also benefit from cleaner SFT targets."
        )
    else:
        lines.append(
            "- On these artifacts, filtering reduced peak validation accuracy slightly, so the main value of the filtered set would need to come from downstream stability or RL warm-start behavior rather than raw SFT accuracy."
        )
    return "\n".join(lines)


def build_markdown(
    dataset_runs: list[RunSummary],
    hparam_runs: list[RunSummary],
    filtered_run: RunSummary,
    normalized_baseline: RunSummary | None,
    dataset_plot_name: str,
    hparam_plot_name: str,
    filtered_plot_name: str,
) -> str:
    best_hparam_run = max(hparam_runs, key=lambda run: run.best_accuracy)
    best_dataset_run = max(dataset_runs, key=lambda run: run.best_accuracy)
    full_hparam_run = next(run for run in hparam_runs if run.label == "2e-5")
    dataset_rows = "\n".join(
        f"| {run.label} | {run.summary['num_train_examples']} | {run.best_accuracy:.4f} | {run.final_accuracy:.4f} | {run.best_eval_step} |"
        for run in dataset_runs
    )
    hparam_rows = "\n".join(
        f"| {run.label} | {run.best_accuracy:.4f} | {run.final_accuracy:.4f} | {run.best_eval_step} |"
        for run in sorted(hparam_runs, key=lambda run: run.best_accuracy, reverse=True)
    )
    commentary = build_commentary(
        dataset_runs, hparam_runs, filtered_run, best_hparam_run
    )
    appendix = ""
    if normalized_baseline is not None:
        appendix = f"""

## Appendix: Normalization Comparison

| Setting | Best Accuracy | Final Accuracy |
| --- | ---: | ---: |
| Full data, `2e-5`, response-token normalization on | {normalized_baseline.best_accuracy:.4f} | {normalized_baseline.final_accuracy:.4f} |
| Full data, `2e-5`, fixed `normalize_constant = 1.0` | {full_hparam_run.best_accuracy:.4f} | {full_hparam_run.final_accuracy:.4f} |
"""
    return f"""# Section 4 SFT Comparison

## Summary

- Best overall run: `{best_hparam_run.label}` on full unfiltered data
- Best overall accuracy: `{best_hparam_run.best_accuracy:.4f}`
- Best checkpoint: step `{best_hparam_run.best_eval_step}` at epoch `{best_hparam_run.best_epoch_fraction:.2f}`
- Best checkpoint path: `{best_hparam_run.summary["best_model_path"]}`
- Final accuracy for that run: `{best_hparam_run.final_accuracy:.4f}`

## Auto Commentary

{commentary}

## Dataset Size Sweep

![Validation accuracy by dataset size]({dataset_plot_name})

| Dataset Label | Train Examples | Best Accuracy | Final Accuracy | Best Step |
| --- | ---: | ---: | ---: | ---: |
{dataset_rows}

The best dataset-size run among the fixed-`2e-5` serial sweep was `{best_dataset_run.label}` with best accuracy `{best_dataset_run.best_accuracy:.4f}`.

## Learning Rate Sweep On Full Data

![Validation accuracy by learning rate]({hparam_plot_name})

| Learning Rate | Best Accuracy | Final Accuracy | Best Step |
| --- | ---: | ---: | ---: |
{hparam_rows}

## Filtered Full Dataset Experiment

Filtered dataset size: `{filtered_run.summary["num_train_examples"]}` examples.

![Filtered vs unfiltered full-data validation accuracy]({filtered_plot_name})

| Setting | Train Examples | Best Accuracy | Final Accuracy | Best Step |
| --- | ---: | ---: | ---: | ---: |
| Unfiltered full data, `2e-5` | {full_hparam_run.summary["num_train_examples"]} | {full_hparam_run.best_accuracy:.4f} | {full_hparam_run.final_accuracy:.4f} | {full_hparam_run.best_eval_step} |
| Filtered full data, `2e-5` | {filtered_run.summary["num_train_examples"]} | {filtered_run.best_accuracy:.4f} | {filtered_run.final_accuracy:.4f} | {filtered_run.best_eval_step} |

Compared to the previous full-data SFT experiment, the filtered-full run changed peak accuracy by `{filtered_run.best_accuracy - full_hparam_run.best_accuracy:.4f}`.
{appendix}
"""


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    dataset_runs = [
        load_run_summary(
            SERIAL_RUN_DIR / "128" / "unfiltered_128" / "summary.json", "128"
        ),
        load_run_summary(
            SERIAL_RUN_DIR / "256" / "unfiltered_256" / "summary.json", "256"
        ),
        load_run_summary(
            SERIAL_RUN_DIR / "512" / "unfiltered_512" / "summary.json", "512"
        ),
        load_run_summary(
            SERIAL_RUN_DIR / "1024" / "unfiltered_1024" / "summary.json", "1024"
        ),
        load_run_summary(
            SERIAL_RUN_DIR / "full" / "unfiltered_full" / "summary.json", "full"
        ),
    ]
    hparam_runs = [
        load_run_summary(
            HPARAM_RUN_DIR / "lr_5em6" / "unfiltered_full" / "summary.json", "5e-6"
        ),
        load_run_summary(
            HPARAM_RUN_DIR / "lr_1em5" / "unfiltered_full" / "summary.json", "1e-5"
        ),
        load_run_summary(
            HPARAM_RUN_DIR / "lr_2em5" / "unfiltered_full" / "summary.json", "2e-5"
        ),
        load_run_summary(
            HPARAM_RUN_DIR / "lr_3em5" / "unfiltered_full" / "summary.json", "3e-5"
        ),
        load_run_summary(
            HPARAM_RUN_DIR / "lr_5em5" / "unfiltered_full" / "summary.json", "5e-5"
        ),
    ]
    filtered_run = load_run_summary(FILTERED_RUN_DIR / "summary.json", "filtered_full")
    normalized_baseline = maybe_load_run_summary(
        DATA_ROOT
        / "section4_sft_serial_20260422-231328_3ep"
        / "full"
        / "unfiltered_full"
        / "summary.json",
        "normalized-2e-5",
    )

    dataset_plot_path = OUTPUT_DIR / "validation_accuracy_by_dataset_size.png"
    hparam_plot_path = OUTPUT_DIR / "validation_accuracy_by_learning_rate_full.png"
    filtered_plot_path = (
        OUTPUT_DIR / "validation_accuracy_filtered_vs_unfiltered_full.png"
    )
    markdown_path = OUTPUT_DIR / "section4_sft_comparison.md"

    render_line_chart(
        title="Validation Accuracy by Dataset Size",
        x_label="Training Epoch",
        y_label="Validation Accuracy",
        series=build_series(
            dataset_runs,
            lambda run: f"{run.label} examples",
            x_fn=lambda run, item: float(item["eval_step"]) / run.steps_per_epoch,
        ),
        output_path=dataset_plot_path,
    )
    render_line_chart(
        title="Validation Accuracy by Learning Rate on Full Data",
        x_label="Eval Step",
        y_label="Validation Accuracy",
        series=build_series(hparam_runs, lambda run: f"lr={run.label}"),
        output_path=hparam_plot_path,
    )
    render_line_chart(
        title="Filtered vs Unfiltered Full-Data Validation Accuracy",
        x_label="Eval Step",
        y_label="Validation Accuracy",
        series=build_series(
            [next(run for run in hparam_runs if run.label == "2e-5"), filtered_run],
            lambda run: (
                "filtered full"
                if run.label == "filtered_full"
                else "unfiltered full (2e-5)"
            ),
        ),
        output_path=filtered_plot_path,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        build_markdown(
            dataset_runs=dataset_runs,
            hparam_runs=hparam_runs,
            filtered_run=filtered_run,
            normalized_baseline=normalized_baseline,
            dataset_plot_name=dataset_plot_path.name,
            hparam_plot_name=hparam_plot_path.name,
            filtered_plot_name=filtered_plot_path.name,
        )
    )

    print(
        json.dumps(
            {
                "markdown_path": str(markdown_path.relative_to(ROOT)),
                "dataset_plot_path": str(dataset_plot_path.relative_to(ROOT)),
                "hparam_plot_path": str(hparam_plot_path.relative_to(ROOT)),
                "filtered_plot_path": str(filtered_plot_path.relative_to(ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
