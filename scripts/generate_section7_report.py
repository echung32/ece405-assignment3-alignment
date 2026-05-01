from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data/section7/grpo_experiment"
DEFAULT_OUTPUT_ROOT = ROOT / "data/section7/analysis"
REPO_NAME = ROOT.name
GENERATION_LOG_RE = re.compile(r"generation_logs_step_(\d+)\.jsonl$")
EXPERIMENT_TITLES = {
    "grpo_train_loop": "GRPO Train Loop",
    "grpo_learning_rate": "GRPO Learning Rate",
    "grpo_baselines": "GRPO Baselines",
    "grpo_length_normalization": "GRPO Length Normalization",
    "grpo_group_standard_deviation": "GRPO Group Standard Deviation",
    "grpo_off_policy_sweep": "GRPO Off-Policy Sweep",
    "grpo_off_policy_clip_ablation": "GRPO Off-Policy Clip Ablation",
    "grpo_prompt_ablation": "GRPO Prompt Ablation",
    "generic": "GRPO Experiment",
}


@dataclass
class RunSummary:
    label: str
    summary_path: Path
    summary: dict
    eval_history: list[dict]
    step_summaries: list[dict]
    generation_logs: list[Path]

    @property
    def best_accuracy(self) -> float:
        return float(self.summary["best_accuracy"])

    @property
    def final_accuracy(self) -> float:
        return float(self.summary["final_accuracy"])

    @property
    def best_eval_step(self) -> int | None:
        value = self.summary.get("best_eval_step")
        return None if value is None else int(value)

    @property
    def wall_clock_seconds(self) -> float | None:
        value = self.summary.get("wall_clock_seconds")
        return None if value is None else float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Section 7 GRPO comparison artifacts."
    )
    parser.add_argument(
        "--campaign",
        nargs="+",
        required=True,
        help="One or more campaign directory names under data/section7/grpo_experiment.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--report-name",
        default=None,
        help="Optional output directory name under data/section7/analysis.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def resolve_artifact_path(path_str: str) -> Path:
    path = Path(path_str)
    candidates: list[Path] = []

    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(ROOT / path)

    parts = path.parts
    for anchor in ("data", "logs", "cs336_alignment", "scripts"):
        if anchor in parts:
            anchor_index = parts.index(anchor)
            candidates.append(ROOT.joinpath(*parts[anchor_index:]))

    if REPO_NAME in parts:
        repo_index = parts.index(REPO_NAME)
        if repo_index + 1 < len(parts):
            candidates.append(ROOT.joinpath(*parts[repo_index + 1 :]))

    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Unable to resolve artifact path: {path_str}")


def extract_generation_step(path: Path) -> int:
    match = GENERATION_LOG_RE.search(path.name)
    return -1 if match is None else int(match.group(1))


def load_run_summary(summary_path: Path, label: str) -> RunSummary:
    summary = load_json(summary_path)
    eval_history = load_jsonl(resolve_artifact_path(summary["eval_history_path"]))
    step_summaries = load_jsonl(resolve_artifact_path(summary["step_summaries_path"]))
    generation_logs = sorted(
        summary_path.parent.glob("generation_logs_step_*.jsonl"),
        key=extract_generation_step,
    )
    return RunSummary(
        label=label,
        summary_path=summary_path,
        summary=summary,
        eval_history=eval_history,
        step_summaries=step_summaries,
        generation_logs=generation_logs,
    )


def discover_runs(campaign_dir: Path) -> list[RunSummary]:
    runs = [
        load_run_summary(path, path.parent.name)
        for path in sorted(campaign_dir.glob("*/summary.json"))
    ]
    if not runs:
        raise FileNotFoundError(
            f"No Section 7 run summaries found under {campaign_dir}"
        )
    return runs


def discover_runs_for_campaigns(
    data_root: Path, campaigns: list[str]
) -> list[RunSummary]:
    runs: list[RunSummary] = []
    for campaign in campaigns:
        runs.extend(discover_runs(data_root / campaign))
    return runs


def infer_experiment_key(campaigns: list[str], report_name: str) -> str:
    normalized = " ".join([*campaigns, report_name]).lower()
    if "expanded50" in normalized or "train_loop" in normalized:
        return "grpo_train_loop"
    if "learning_rate" in normalized:
        return "grpo_learning_rate"
    if "baseline" in normalized:
        return "grpo_baselines"
    if "lengthnorm" in normalized or "length_normalization" in normalized:
        return "grpo_length_normalization"
    if "stdnorm" in normalized or "standard_deviation" in normalized:
        return "grpo_group_standard_deviation"
    if "no_clip" in normalized or "clip_ablation" in normalized:
        return "grpo_off_policy_clip_ablation"
    if "offpolicy" in normalized or "off_policy" in normalized:
        return "grpo_off_policy_sweep"
    if "prompt" in normalized:
        return "grpo_prompt_ablation"
    return "generic"


def format_optional_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def format_wall_clock_minutes(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    return f"{seconds / 60.0:.1f}"


def get_last_numeric(items: list[dict], key: str) -> float | None:
    for item in reversed(items):
        value = item.get(key)
        if value is not None:
            return float(value)
    return None


def get_max_numeric(items: list[dict], key: str) -> float | None:
    values = [float(item[key]) for item in items if item.get(key) is not None]
    if not values:
        return None
    return max(values)


def get_mean_numeric(items: list[dict], key: str) -> float | None:
    values = [float(item[key]) for item in items if item.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def ordered_runs(runs: list[RunSummary]) -> list[RunSummary]:
    return sorted(runs, key=lambda run: (-run.best_accuracy, run.label))


def format_learning_rate(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.0e}"
    return str(value)


def pretty_loss_type(value: object) -> str:
    mapping = {
        "no_baseline": "No baseline",
        "reinforce_with_baseline": "Baseline",
        "grpo_clip": "GRPO-Clip",
        "grpo_no_clip": "GRPO-No-Clip",
    }
    return mapping.get(str(value), str(value))


def display_label(
    run: RunSummary, experiment_key: str, include_campaign_scope: bool = False
) -> str:
    if experiment_key == "grpo_learning_rate":
        return f"lr={format_learning_rate(run.summary.get('learning_rate'))}"
    if experiment_key == "grpo_baselines":
        return pretty_loss_type(run.summary.get("loss_type"))
    if experiment_key == "grpo_length_normalization":
        norm = run.summary.get("length_normalization", "masked_mean")
        constant = run.summary.get("length_normalize_constant")
        if constant is None or norm == "masked_mean":
            return str(norm)
        return f"{norm} ({int(float(constant))})"
    if experiment_key == "grpo_group_standard_deviation":
        return f"std={run.summary.get('use_std_normalization')}"
    if experiment_key == "grpo_off_policy_sweep":
        label = (
            f"ep={run.summary.get('epochs_per_rollout_batch')}, "
            f"tb={run.summary.get('train_batch_size')}"
        )
        if not include_campaign_scope:
            return label
        campaign_name = run.summary_path.parent.parent.name
        if "offpolicy_broad" in campaign_name:
            return f"{label} (broad)"
        if "offpolicy_focused" in campaign_name:
            return f"{label} (focused)"
        return label
    if experiment_key == "grpo_off_policy_clip_ablation":
        return pretty_loss_type(run.summary.get("loss_type"))
    if experiment_key == "grpo_prompt_ablation":
        reward_function = run.summary.get("reward_function", "r1_zero")
        return "Question-only" if reward_function == "question_only" else "R1-Zero"
    if experiment_key == "grpo_train_loop":
        return f"{pretty_loss_type(run.summary.get('loss_type'))}, lr={format_learning_rate(run.summary.get('learning_rate'))}"
    return run.label


def build_series(
    runs: list[RunSummary],
    experiment_key: str,
    source: str,
    x_key: str,
    value_key: str,
    x_transform: Callable[[float], float] | None = None,
) -> list[tuple[str, list[tuple[float, float]], tuple[float, float, float, float]]]:
    transform = x_transform or (lambda value: value)
    cmap = plt.get_cmap("tab10")
    series = []
    include_campaign_scope = (
        len({run.summary_path.parent.parent.name for run in runs}) > 1
    )
    for index, run in enumerate(ordered_runs(runs)):
        items = run.eval_history if source == "eval" else run.step_summaries
        points = []
        for item in items:
            if item.get(x_key) is None or item.get(value_key) is None:
                continue
            points.append((transform(float(item[x_key])), float(item[value_key])))
        if points:
            series.append(
                (
                    display_label(run, experiment_key, include_campaign_scope),
                    points,
                    cmap(index % 10),
                )
            )
    return series


def render_panel_chart(
    ax: plt.Axes,
    title: str,
    x_label: str,
    y_label: str,
    series: list[
        tuple[str, list[tuple[float, float]], tuple[float, float, float, float]]
    ],
) -> None:
    if not series:
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    for label, points, color in series:
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        ax.plot(
            x_values,
            y_values,
            label=label,
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=3.2,
            alpha=0.95,
        )

    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.margins(x=0.02)
    ax.legend(fontsize=8, frameon=False, loc="best")


def save_panel_figure(
    chart_specs: list[
        tuple[
            str,
            str,
            str,
            list[
                tuple[str, list[tuple[float, float]], tuple[float, float, float, float]]
            ],
        ]
    ],
    output_path: Path,
    figure_title: str,
) -> None:
    ncols = 1 if len(chart_specs) == 1 else 2
    nrows = math.ceil(len(chart_specs) / ncols)
    figure, axes = plt.subplots(
        nrows, ncols, figsize=(8.2 * ncols, 5.8 * nrows), dpi=180
    )
    if not isinstance(axes, (list, tuple)):
        try:
            flat_axes = list(axes.flat)
        except AttributeError:
            flat_axes = [axes]
    else:
        flat_axes = list(axes)
    for ax, (title, x_label, y_label, series) in zip(
        flat_axes, chart_specs, strict=True
    ):
        render_panel_chart(ax, title, x_label, y_label, series)
    for ax in flat_axes[len(chart_specs) :]:
        ax.set_axis_off()
    figure.suptitle(figure_title, fontsize=14, weight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def truncate_text(text: str, limit: int = 280) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3]}..."


def summarize_run(run: RunSummary) -> dict[str, str]:
    return {
        "best_accuracy": format_optional_float(run.best_accuracy),
        "final_accuracy": format_optional_float(run.final_accuracy),
        "best_eval_step": "n/a"
        if run.best_eval_step is None
        else str(run.best_eval_step),
        "loss_type": str(run.summary.get("loss_type", "n/a")),
        "reward_function": str(run.summary.get("reward_function", "r1_zero")),
        "length_normalization": str(
            run.summary.get("length_normalization", "masked_mean")
        ),
        "length_normalize_constant": str(
            run.summary.get("length_normalize_constant", "n/a")
        ),
        "use_std_normalization": str(run.summary.get("use_std_normalization", "n/a")),
        "epochs_per_rollout_batch": str(
            run.summary.get("epochs_per_rollout_batch", "n/a")
        ),
        "train_batch_size": str(run.summary.get("train_batch_size", "n/a")),
        "wall_clock_minutes": format_wall_clock_minutes(run.wall_clock_seconds),
        "peak_mean_reward": format_optional_float(
            get_max_numeric(run.step_summaries, "mean_reward")
        ),
        "final_mean_reward": format_optional_float(
            get_last_numeric(run.step_summaries, "mean_reward")
        ),
        "avg_response_length": format_optional_float(
            get_mean_numeric(run.step_summaries, "mean_response_length"), digits=1
        ),
        "final_response_length": format_optional_float(
            get_last_numeric(run.step_summaries, "mean_response_length"), digits=1
        ),
        "peak_format_reward": format_optional_float(
            get_max_numeric(run.step_summaries, "mean_format_reward")
        ),
    }


def summarize_groups(
    runs: list[RunSummary], label_fn: Callable[[RunSummary], str]
) -> str:
    grouped: dict[str, list[RunSummary]] = {}
    for run in runs:
        grouped.setdefault(label_fn(run), []).append(run)
    lines = []
    for group_label, group_runs in sorted(grouped.items()):
        best_run = max(group_runs, key=lambda run: run.best_accuracy)
        lines.append(
            f"- `{group_label}`: best run `{best_run.label}` reached accuracy {best_run.best_accuracy:.4f} and peak rollout reward {format_optional_float(get_max_numeric(best_run.step_summaries, 'mean_reward'))}"
        )
    return "\n".join(lines)


def select_example_logs(run: RunSummary) -> list[Path]:
    if not run.generation_logs:
        return []
    indices = [0, len(run.generation_logs) // 2, len(run.generation_logs) - 1]
    selected: list[Path] = []
    seen: set[Path] = set()
    for index in indices:
        path = run.generation_logs[index]
        if path in seen:
            continue
        seen.add(path)
        selected.append(path)
    return selected


def choose_example_entry(entries: list[dict]) -> dict | None:
    if not entries:
        return None
    for entry in entries:
        reward = entry.get("reward", {}).get("reward")
        if reward == 1.0:
            return entry
    return entries[0]


def build_rollout_examples(run: RunSummary) -> str:
    if not run.generation_logs:
        return "No generation logs were found for the selected run."

    sections = []
    for path in select_example_logs(run):
        entries = load_jsonl(path)
        example = choose_example_entry(entries)
        if example is None:
            continue
        reward = example.get("reward", {})
        sections.append(
            "\n".join(
                [
                    f"### Step {extract_generation_step(path)}",
                    f"- Reward tuple: total={reward.get('reward', 'n/a')}, format={reward.get('format_reward', 'n/a')}, answer={reward.get('answer_reward', 'n/a')}",
                    f"- Problem excerpt: {truncate_text(example.get('problem', ''), 220)}",
                    f"- Response excerpt: {truncate_text(example.get('response', ''), 320)}",
                ]
            )
        )
    if not sections:
        return "Generation logs were present but no entries could be parsed."
    return "\n\n".join(sections)


def build_experiment_notes(
    experiment_key: str, runs: list[RunSummary], wall_clock_path: str | None
) -> str:
    if experiment_key == "grpo_learning_rate":
        return summarize_groups(
            runs, lambda run: f"lr={run.summary.get('learning_rate')}"
        )
    if experiment_key == "grpo_baselines":
        return summarize_groups(
            runs, lambda run: f"loss_type={run.summary.get('loss_type')}"
        )
    if experiment_key == "grpo_length_normalization":
        return summarize_groups(
            runs,
            lambda run: (
                f"length_norm={run.summary.get('length_normalization')}, "
                f"constant={run.summary.get('length_normalize_constant', 'n/a')}"
            ),
        )
    if experiment_key == "grpo_group_standard_deviation":
        return summarize_groups(
            runs,
            lambda run: (
                f"use_std_normalization={run.summary.get('use_std_normalization')}"
            ),
        )
    if experiment_key == "grpo_off_policy_sweep":
        return summarize_groups(
            runs,
            lambda run: (
                f"epochs={run.summary.get('epochs_per_rollout_batch')}, "
                f"train_batch={run.summary.get('train_batch_size')}"
            ),
        )
    if experiment_key == "grpo_off_policy_clip_ablation":
        return summarize_groups(
            runs, lambda run: f"loss_type={run.summary.get('loss_type')}"
        )
    if experiment_key == "grpo_prompt_ablation":
        return summarize_groups(
            runs,
            lambda run: (
                f"reward_function={run.summary.get('reward_function', 'r1_zero')}"
            ),
        )
    if experiment_key == "grpo_train_loop":
        best_run = max(runs, key=lambda run: run.best_accuracy)
        return f"- Best run for rollout examples: `{best_run.label}`\n\n{build_rollout_examples(best_run)}"

    available_fields = sorted(
        {
            key
            for run in runs
            for item in run.step_summaries[:5]
            for key in item.keys()
            if not isinstance(item.get(key), dict)
        }
    )
    return f"- Available step-summary fields: {', '.join(available_fields)}"


def build_commentary(experiment_key: str, runs: list[RunSummary]) -> str:
    ranked_runs = ordered_runs(runs)
    best_run = ranked_runs[0]
    lines: list[str] = []

    if len(ranked_runs) > 1:
        runner_up = ranked_runs[1]
        delta = best_run.best_accuracy - runner_up.best_accuracy
        lines.append(
            f"- Best observed run was `{best_run.label}` at {best_run.best_accuracy:.4f} validation accuracy, ahead of `{runner_up.label}` by {delta:.4f}."
        )

    best_drop = best_run.best_accuracy - best_run.final_accuracy
    if best_drop > 0.02:
        lines.append(
            f"- The best checkpoint for `{best_run.label}` was meaningfully ahead of its final checkpoint by {best_drop:.4f}, which suggests late-run instability or overtraining."
        )
    else:
        lines.append(
            f"- `{best_run.label}` stayed stable through the end of training, with only {best_drop:.4f} difference between best and final validation accuracy."
        )

    if experiment_key == "grpo_learning_rate":
        divergent_runs = [run for run in ranked_runs if run.final_accuracy < 0.05]
        if divergent_runs:
            lines.append(
                f"- Higher learning-rate settings were unstable here; `{divergent_runs[0].label}` collapsed to final accuracy {divergent_runs[0].final_accuracy:.4f}."
            )
    elif experiment_key == "grpo_baselines":
        baseline_runs = {run.summary.get("loss_type"): run for run in runs}
        no_baseline = baseline_runs.get("no_baseline")
        with_baseline = baseline_runs.get("reinforce_with_baseline")
        if no_baseline is not None and with_baseline is not None:
            diff = no_baseline.best_accuracy - with_baseline.best_accuracy
            lines.append(
                f"- In this campaign, `no_baseline` outperformed `reinforce_with_baseline` by {diff:.4f} best validation accuracy, while the baseline run held a slightly higher peak rollout reward."
            )
    elif experiment_key == "grpo_off_policy_sweep":
        fastest_runs = [
            run for run in ranked_runs if run.wall_clock_seconds is not None
        ]
        if fastest_runs:
            fastest_run = min(
                fastest_runs, key=lambda run: run.wall_clock_seconds or float("inf")
            )
            lines.append(
                f"- The fastest run was `{fastest_run.label}` at {format_wall_clock_minutes(fastest_run.wall_clock_seconds)} minutes, while the best-accuracy run took {format_wall_clock_minutes(best_run.wall_clock_seconds)} minutes."
            )
    elif experiment_key == "grpo_prompt_ablation":
        prompt_runs = {
            run.summary.get("reward_function", "r1_zero"): run for run in runs
        }
        r1_zero_run = prompt_runs.get("r1_zero")
        question_only_run = prompt_runs.get("question_only")
        if r1_zero_run is not None and question_only_run is not None:
            diff = question_only_run.best_accuracy - r1_zero_run.best_accuracy
            lines.append(
                f"- The question-only setup beat the R1-Zero setup by {diff:.4f} best validation accuracy in this artifact slice."
            )

    return "\n".join(lines)


def build_markdown(
    campaigns: list[str],
    report_name: str,
    experiment_key: str,
    runs: list[RunSummary],
    artifact_paths: list[str],
    wall_clock_path: str | None,
) -> str:
    best_run = max(runs, key=lambda run: run.best_accuracy)
    campaign_lines = "\n".join(f"- `{campaign}`" for campaign in campaigns)
    artifact_lines = "\n".join(f"- `{artifact}`" for artifact in artifact_paths)
    rows = []
    for run in ordered_runs(runs):
        metrics = summarize_run(run)
        rows.append(
            "| {label} | {best_accuracy} | {final_accuracy} | {peak_mean_reward} | {final_mean_reward} | {avg_response_length} | {loss_type} | {reward_function} | {length_normalization} | {use_std_normalization} | {epochs_per_rollout_batch} | {train_batch_size} | {wall_clock_minutes} |".format(
                label=run.label,
                **metrics,
            )
        )
    run_table = "\n".join(rows)
    notes = build_experiment_notes(experiment_key, runs, wall_clock_path)
    commentary = build_commentary(experiment_key, runs)
    return f"""# {EXPERIMENT_TITLES.get(experiment_key, EXPERIMENT_TITLES["generic"])} Analysis

Report name:
- `{report_name}`

Campaigns:
{campaign_lines}

Summary:
- Best run: `{best_run.label}`
- Best validation accuracy: `{best_run.best_accuracy:.4f}`
- Final validation accuracy for best run: `{best_run.final_accuracy:.4f}`

Generated artifacts:
{artifact_lines}

## Run Table

| Run | Best Accuracy | Final Accuracy | Peak Reward | Final Reward | Avg Response Length | Loss Type | Reward Fn | Length Norm | Std Norm | Epochs | Train Batch | Wall Clock (min) |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: |
{run_table}

## Figures

![Section 7 combined metrics](section7_combined_metrics.png)

## Auto Commentary

{commentary}

## Deliverable Notes

{notes}
"""


def build_placeholder(experiment_key: str, artifact_paths: list[str]) -> str:
    artifact_lines = "\n".join(f"- `{artifact}`" for artifact in artifact_paths)
    sections = {
        "grpo_learning_rate": "Summarize the learning-rate sweep, identify the winning learning rate, and add a short note on any reward or response-length trends.",
        "grpo_baselines": "Compare `no_baseline` against `reinforce_with_baseline` using the run table and the combined metrics figure.",
        "grpo_length_normalization": "Compare `masked_mean` against constant-normalized `masked_normalize` and note any stability differences visible in reward and response length.",
        "grpo_group_standard_deviation": "Compare `use_std_normalization=True` against `False` using the generated figure and run table.",
        "grpo_off_policy_sweep": "Use the combined metrics chart, which includes both step-based and wall-clock panels, to discuss this off-policy sweep.",
        "grpo_off_policy_clip_ablation": "Compare `grpo_clip` against the unclipped variant using reward curves and the per-run summary table.",
        "grpo_prompt_ablation": "Compare the prompt and reward-function variants using the combined metrics figure and response-length summaries.",
        "grpo_train_loop": "Use the rollout examples and the combined metrics chart to illustrate qualitative changes over training.",
        "generic": "Use the run table and figures above to draft the final writeup.",
    }
    section_text = sections.get(experiment_key, sections["generic"])
    return f"""# Section 7 Deliverables Placeholder

Relevant artifacts:
{artifact_lines}

What to write:
- {section_text}
"""


def build_wall_clock_chart_specs(
    runs: list[RunSummary], experiment_key: str
) -> list[
    tuple[
        str,
        str,
        str,
        list[tuple[str, list[tuple[float, float]], tuple[float, float, float, float]]],
    ]
]:
    return [
        (
            "Validation Accuracy vs Elapsed Minutes",
            "Elapsed Minutes",
            "Validation Accuracy",
            build_series(
                runs,
                experiment_key=experiment_key,
                source="eval",
                x_key="elapsed_seconds",
                value_key="accuracy",
                x_transform=lambda value: value / 60.0,
            ),
        ),
        (
            "Rollout Reward vs Elapsed Minutes",
            "Elapsed Minutes",
            "Mean Rollout Reward",
            build_series(
                runs,
                experiment_key=experiment_key,
                source="step",
                x_key="elapsed_seconds",
                value_key="mean_reward",
                x_transform=lambda value: value / 60.0,
            ),
        ),
    ]


def main() -> None:
    args = parse_args()
    report_name = args.report_name
    if report_name is None:
        report_name = (
            args.campaign[0]
            if len(args.campaign) == 1
            else "__vs__".join(args.campaign)
        )

    output_dir = args.output_root / report_name
    runs = discover_runs_for_campaigns(args.data_root, args.campaign)
    experiment_key = infer_experiment_key(args.campaign, report_name)

    plt.style.use("seaborn-v0_8-whitegrid")

    artifact_paths: list[str] = []
    metrics_path = output_dir / "section7_combined_metrics.png"
    chart_specs = [
        (
            "Validation Accuracy vs Step",
            "GRPO Step",
            "Validation Accuracy",
            build_series(
                runs,
                experiment_key=experiment_key,
                source="eval",
                x_key="eval_step",
                value_key="accuracy",
            ),
        ),
        (
            "Rollout Reward vs Step",
            "GRPO Step",
            "Mean Rollout Reward",
            build_series(
                runs,
                experiment_key=experiment_key,
                source="step",
                x_key="grpo_step",
                value_key="mean_reward",
            ),
        ),
    ]
    if experiment_key == "grpo_off_policy_sweep":
        chart_specs.extend(build_wall_clock_chart_specs(runs, experiment_key))
    save_panel_figure(
        chart_specs=chart_specs,
        output_path=metrics_path,
        figure_title=f"{EXPERIMENT_TITLES.get(experiment_key, EXPERIMENT_TITLES['generic'])}: Metrics",
    )
    artifact_paths.append(metrics_path.name)

    wall_clock_path: str | None = None

    markdown_path = output_dir / "section7_grpo_comparison.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        build_markdown(
            campaigns=args.campaign,
            report_name=report_name,
            experiment_key=experiment_key,
            runs=runs,
            artifact_paths=artifact_paths,
            wall_clock_path=wall_clock_path,
        )
    )
    artifact_paths.append(markdown_path.name)

    placeholder_path = output_dir / "section7_deliverables_placeholder.md"
    placeholder_path.write_text(build_placeholder(experiment_key, artifact_paths))


if __name__ == "__main__":
    main()
