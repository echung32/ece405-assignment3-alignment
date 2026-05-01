from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = (
    ROOT
    / "data"
    / "section5"
    / "expert_iteration"
    / "section5_ei_full_20260426_fix_sweep"
)
OUTPUT_DIR = ROOT / "data" / "section5" / "analysis" / CAMPAIGN_DIR.name
SECTION4_HPARAM_DIR = (
    ROOT
    / "data"
    / "section4"
    / "sft_experiment"
    / "section4_sft_all_20260423-0010_norm1_hparam"
)
STYLE = "seaborn-v0_8-whitegrid"
COLORS = [
    "#0b6e4f",
    "#d17b0f",
    "#6c5ce7",
    "#c0392b",
    "#2980b9",
    "#7f8c8d",
]
BASELINE_COLOR = "#111111"


@dataclass
class RunSummary:
    label: str
    summary_path: Path
    summary: dict
    eval_history: list[dict]

    @property
    def best_accuracy(self) -> float:
        return float(self.summary.get("best_accuracy", 0.0))

    @property
    def final_accuracy(self) -> float:
        if not self.eval_history:
            return 0.0
        return float(self.eval_history[-1]["accuracy"])

    @property
    def best_eval_step(self) -> int:
        return int(self.summary.get("best_eval_step", 0))

    @property
    def entropy_at_best(self) -> float:
        if not self.eval_history:
            return 0.0
        best_entry = max(self.eval_history, key=lambda entry: float(entry["accuracy"]))
        return float(
            best_entry.get("mean_token_entropy", best_entry.get("format_entropy", 0.0))
        )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_run_summary(summary_path: Path) -> RunSummary:
    summary = load_json(summary_path)
    eval_history = load_jsonl(summary_path.with_name("eval_history.jsonl"))
    relative = summary_path.parent.relative_to(CAMPAIGN_DIR)
    return RunSummary(
        label=display_label(relative),
        summary_path=summary_path,
        summary=summary,
        eval_history=eval_history,
    )


def discover_runs() -> list[RunSummary]:
    runs: list[RunSummary] = []
    for summary_path in sorted(CAMPAIGN_DIR.glob("*/summary.json")):
        runs.append(load_run_summary(summary_path))
    if not runs:
        raise FileNotFoundError(f"No EI runs found under {CAMPAIGN_DIR}")
    return runs


def display_label(relative_dir: Path) -> str:
    parts = relative_dir.name.split("_")
    formatted: list[str] = []
    for part in parts:
        if part.startswith("g") and part[1:].isdigit():
            formatted.append(f"group {part[1:]}")
        elif part.startswith("db") and part[2:].isdigit():
            formatted.append(f"db {part[2:]}")
        elif part.startswith("ep") and part[2:].isdigit():
            formatted.append(f"epochs {part[2:]}")
        elif part.startswith("steps") and part[5:].isdigit():
            formatted.append(f"EI {part[5:]}")
        else:
            formatted.append(part)
    return ", ".join(formatted)


def build_accuracy_series(run: RunSummary) -> list[tuple[int, float]]:
    return [
        (int(entry["ei_step"]), float(entry["accuracy"])) for entry in run.eval_history
    ]


def build_entropy_series(run: RunSummary) -> list[tuple[int, float]]:
    return [
        (
            int(entry["ei_step"]),
            float(entry.get("mean_token_entropy", entry.get("format_entropy", 0.0))),
        )
        for entry in run.eval_history
    ]


def render_line_chart(
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: list[tuple[str, list[tuple[int, float]], str]],
    output_path: Path,
    baseline: tuple[str, float] | None = None,
) -> None:
    with plt.style.context(STYLE):
        figure, ax = plt.subplots(figsize=(10.4, 6.0), dpi=180)
        for label, points, color in series:
            x_values = [point[0] for point in points]
            y_values = [point[1] for point in points]
            ax.plot(
                x_values,
                y_values,
                label=label,
                linewidth=2.2,
                marker="o",
                markersize=4.0,
                color=color,
            )

        if baseline is not None:
            baseline_label, baseline_value = baseline
            ax.axhline(
                baseline_value,
                color=BASELINE_COLOR,
                linestyle="--",
                linewidth=1.6,
                label=f"{baseline_label} ({baseline_value:.4f})",
            )

        ax.set_title(title, fontsize=13, weight="bold")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.margins(x=0.03)
        figure.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, bbox_inches="tight")
        plt.close(figure)


def load_best_section4_sft_run() -> RunSummary:
    runs = [
        load_section4_run(summary_path)
        for summary_path in sorted(
            SECTION4_HPARAM_DIR.glob("*/unfiltered_full/summary.json")
        )
    ]
    if not runs:
        raise FileNotFoundError(
            f"No Section 4 SFT summaries found under {SECTION4_HPARAM_DIR}"
        )
    return max(runs, key=lambda run: run.best_accuracy)


def load_section4_run(summary_path: Path) -> RunSummary:
    summary = load_json(summary_path)
    eval_history = load_jsonl(summary_path.with_name("eval_history.jsonl"))
    return RunSummary(
        label=summary_path.parent.parent.name.replace("lr_", "").replace("em", "e-"),
        summary_path=summary_path,
        summary=summary,
        eval_history=[
            {
                "ei_step": index,
                "accuracy": entry["accuracy"],
                "mean_token_entropy": entry.get(
                    "mean_token_entropy", entry.get("format_entropy", 0.0)
                ),
            }
            for index, entry in enumerate(eval_history)
        ],
    )


def build_markdown(runs: list[RunSummary], sft_baseline: RunSummary) -> str:
    best_run = max(runs, key=lambda run: run.best_accuracy)
    worst_run = min(runs, key=lambda run: run.best_accuracy)
    delta_vs_sft = best_run.best_accuracy - sft_baseline.best_accuracy

    lines = [
        "# Section 5 Expert Iteration Comparison",
        "",
        "## Key Takeaways",
        "",
        f"- The best EI configuration was `{best_run.label}` with best validation accuracy {best_run.best_accuracy:.4f} at EI step {best_run.best_eval_step}.",
        f"- The weakest EI configuration was `{worst_run.label}` with best validation accuracy {worst_run.best_accuracy:.4f}, so the sweep spans {best_run.best_accuracy - worst_run.best_accuracy:.4f} accuracy.",
        f"- Compared against the best Section 4 SFT baseline (`{sft_baseline.label}` at {sft_baseline.best_accuracy:.4f}), the strongest EI run {'improved' if delta_vs_sft >= 0 else 'trailed'} by {abs(delta_vs_sft):.4f}.",
        "",
        "## Required Discussion",
        "",
        f"Relative to the best Section 4 SFT checkpoint, expert iteration {'delivers a small gain' if delta_vs_sft >= 0 else 'does not recover the same accuracy'}: the top EI run reaches {best_run.best_accuracy:.4f} versus {sft_baseline.best_accuracy:.4f} for SFT.",
        f"Across EI steps, the highest-performing configuration peaks at EI step {best_run.best_eval_step}, while later steps do not produce a uniformly monotonic improvement across all settings, so the sweep quality depends on the configuration rather than EI step alone.",
        "",
        "## Run Table",
        "",
        "| Run | Best Accuracy | Final Accuracy | Best EI Step | Entropy At Best |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for run in sorted(runs, key=lambda item: item.best_accuracy, reverse=True):
        lines.append(
            f"| `{run.label}` | {run.best_accuracy:.4f} | {run.final_accuracy:.4f} | {run.best_eval_step} | {run.entropy_at_best:.4f} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    runs = discover_runs()
    sft_baseline = load_best_section4_sft_run()

    accuracy_series = [
        (run.label, build_accuracy_series(run), COLORS[index % len(COLORS)])
        for index, run in enumerate(runs)
    ]
    entropy_series = [
        (run.label, build_entropy_series(run), COLORS[index % len(COLORS)])
        for index, run in enumerate(runs)
    ]

    accuracy_plot_path = OUTPUT_DIR / "validation_accuracy_by_run.png"
    entropy_plot_path = OUTPUT_DIR / "entropy_by_run.png"
    render_line_chart(
        title="Section 5 Validation Accuracy Across EI Steps",
        x_label="EI Step",
        y_label="Validation Accuracy",
        series=accuracy_series,
        output_path=accuracy_plot_path,
        baseline=(
            f"Best Section 4 SFT ({sft_baseline.label})",
            sft_baseline.best_accuracy,
        ),
    )
    render_line_chart(
        title="Section 5 Format Entropy Across EI Steps",
        x_label="EI Step",
        y_label="Format Entropy",
        series=entropy_series,
        output_path=entropy_plot_path,
    )

    markdown_path = OUTPUT_DIR / "section5_expert_iteration_comparison.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(build_markdown(runs, sft_baseline))

    report_summary = {
        "campaign": CAMPAIGN_DIR.name,
        "run_count": len(runs),
        "best_run": max(runs, key=lambda run: run.best_accuracy).label,
        "best_accuracy": max(run.best_accuracy for run in runs),
        "best_section4_sft": sft_baseline.label,
        "best_section4_sft_accuracy": sft_baseline.best_accuracy,
        "accuracy_plot_path": str(accuracy_plot_path.relative_to(ROOT)),
        "entropy_plot_path": str(entropy_plot_path.relative_to(ROOT)),
        "markdown_path": str(markdown_path.relative_to(ROOT)),
    }
    (OUTPUT_DIR / "report_summary.json").write_text(
        json.dumps(report_summary, indent=2) + "\n"
    )
    print(json.dumps(report_summary, indent=2))


if __name__ == "__main__":
    main()
