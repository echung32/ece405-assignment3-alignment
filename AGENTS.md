# AGENTS.md

You are working in the `ece405-assignment3-alignment` repository. Treat this file as operating instructions for the agent.

## Start Of Session

- Read this file before making changes.
- Prefer local evidence from the repo over assumptions.
- Inspect the specific files, scripts, tests, and logs relevant to the current task before editing.
- If you change the workflow, orchestration, logging layout, or output conventions, update this file before you finish.

## Repo Ground Truth

- Core code lives in `cs336_alignment/`.
- Runnable scripts live in `scripts/`.
- Tests live in `tests/`.
- Prompt templates live in `cs336_alignment/prompts/`.
- Generated artifacts belong under `data/` or `logs/`, not the repo root.
- Section 7 GRPO runs should write under `data/section7/grpo_experiment/<campaign>/<run_name>/` and reports under `data/section7/analysis/<campaign>/`.
- Section 7 GRPO `run_name` values now end with a stable short config hash suffix like `_cfg1a2b3c4d` so configs that differ in omitted fields such as `train_batch_size` do not collide on disk.
- Section 7 GRPO sweep entrypoints live under `scripts/slurm/`; the `RUN_CONFIGS` arrays in those Slurm launchers are the source of truth for sweep settings.
- Section 7 throughput tuning probes live at `scripts/run_section7_throughput_probe.sh` and write under `logs/section7/<campaign>/` plus `data/section7/grpo_experiment/<campaign>/` like the main experiments.
- Section 3 baseline launchers live at `scripts/run_section3_math_baseline.sh` and `scripts/run_section3_math_baseline_tmux.sh`; they evaluate on `data/math/val.jsonl` by default and write under `logs/section3/<campaign>/` plus `data/section3/math_baseline/<campaign>/`.
- Section 7 Slurm launchers live under `scripts/slurm/`; they use array jobs to run per-config sweeps in parallel while preserving the same `logs/section7/<campaign>/` and `data/section7/grpo_experiment/<campaign>/` layout as the shell entrypoints.
- Section 7 Slurm launchers reject submission-time extra experiment args such as `--learning-rate`; edit the launcher `RUN_CONFIGS` instead.

## Python And Commands

- Use `uv run ...` for Python commands unless there is a concrete reason not to.
- Use `uv sync` after dependency changes.
- Use `uv run pytest` for the full test suite.
- Use `test_and_make_submission.sh` only when you specifically need the submission helper.
- Preserve `gpu_memory_utilization=0.8` for vLLM scripts unless there is a concrete, validated reason to change it.

## Local Defaults

- Default model path for the main assignment work: `data/Qwen/Qwen2.5-Math-1.5B`.
- Optional extra-credit model paths: `data/Qwen/Qwen2.5-0.5B` and `data/Qwen/Qwen2.5-3B-Instruct`.
- Default MATH splits: `data/math/train.jsonl`, `data/math/val.jsonl`, `data/math/test.jsonl`.
- On the GH200 Section 7 setup, the throughput-tuned on-policy default is `train_batch_size=256` with `gradient_accumulation_steps=64`; when off-policy scripts vary `train_batch_size`, scale `gradient_accumulation_steps` with it to preserve the intended microbatch size.

## Reuse Before Rebuilding

- Reuse grading logic from `cs336_alignment/drgrpo_grader.py` when applicable.
- Reuse existing helpers and experiment utilities before adding new wrappers or duplicate training code.
- Prefer existing entrypoints and orchestration scripts over ad hoc commands when the repo already has a supported workflow.
- For Section 7 experiment ordering, preserve the dependency chain `learning rate -> baselines -> length normalization -> std normalization -> off-policy sweep -> no-clip ablation -> prompt ablation` unless there is a concrete reason to deviate.

## Validation Pipeline

- Start from the smallest discriminating check, not the largest rerun.
- Before broad experiments, prefer one of these cheap validations when applicable:
	- `uv run python -m py_compile <touched_file>`
	- a narrow `uv run pytest ...` target for the touched logic
	- a smoke-test mode on the relevant script
	- replaying the smallest failed artifact slice if debugging runtime or memory issues
- After the first substantive code edit, run one focused validation immediately before doing more broad work.
- If a narrow validation fails, repair that slice and rerun the same validation before expanding scope.
- Use full-suite runs only after the touched slice is locally stable, or when the user explicitly asks for it.
- For Section 7 throughput or memory tuning, prefer a 1-step or 2-step probe before launching a sweep, and capture both the script output and sampled `nvidia-smi` logs.

## Background Run Pipeline

- For long-running training, evaluation, or sweep jobs, prefer a detached `tmux` session rather than tying the run to an interactive shell.
- Before starting a new background run, check whether a session for the same task or campaign is already active.
- Prefer a small smoke test before launching the full detached run.
- When a task has multiple configs, prefer a dedicated orchestrator or sweep script over manual repeated commands.
- If a wrapper script prints identifiers like a `session=` name or a `campaign=` name, capture and reuse them when monitoring logs and artifacts.

## Logging And Output Expectations

- Keep human-readable logs under `logs/<area>/<campaign>/` when the task has a natural area or campaign grouping.
- Keep sweep-level or orchestration progress in an `orchestrator.log` at the campaign root when a run launches multiple configs.
- Keep per-run logs alongside the orchestrator log with descriptive names.
- Keep machine-readable outputs under `data/<area>/<experiment_or_campaign>/<run_name>/`.
 - Keep derived reports and plots under `data/<area>/analysis/<campaign>/` or the matching analysis directory for that task.
- Keep Section 3 baseline machine-readable outputs under `data/section3/math_baseline/<campaign>/` and its human-readable logs under `logs/section3/<campaign>/`.
- Section 7 GRPO step summaries may include `phase_seconds` and `train_cuda_memory_mb` when timing or memory tracing is enabled; treat these as the source of truth for between-phase stalls.

## Monitoring And Debugging

- When monitoring a background run, read `orchestrator.log` first if it exists, then inspect the active per-run log.
- Use on-disk artifacts as the source of truth for where a run stopped.
- If a run dies mid-pipeline, identify the last completed artifact and the first missing expected artifact before hypothesizing about root cause.
- When debugging memory or stability issues, reproduce the smallest failing slice instead of relaunching the whole workflow.
- Suspect cleanup and phase-boundary memory retention when a run consistently survives early iterations and dies later without a Python traceback.

## Reporting And Closeout

- If the repo already has a report-generation script for the task, run or verify it after the experiment finishes.
- Confirm that plots, summaries, and machine-readable outputs were written to the expected analysis directory.
- `scripts/generate_section7_report.py` now uses matplotlib and writes `section7_grpo_comparison.md`, a side-by-side `section7_combined_metrics.png`, and `section7_deliverables_placeholder.md`; off-policy reports fold both step-based and elapsed-time panels into the same combined metrics figure.
- In the final user-facing summary, report what changed, what was validated, what is still running, and any concrete next monitoring step.

## Minimal Startup Prompt For Future Agents

Use the following prompt when bootstrapping a new agent for work in this repo:

```text
Read AGENTS.md first and follow it strictly.

You are working in the ece405-assignment3-alignment repository.

Required startup steps:
1. Read AGENTS.md and inspect the files and scripts relevant to the current task.
2. Check whether a related tmux session or long-running background job is already active before launching anything new.
3. Inspect the relevant log directory and read orchestrator.log first when it exists.
4. Use machine-readable artifacts under data/... and logs under logs/... as the source of truth.
5. Before broad reruns, prefer the smallest discriminating validation:
   - uv run python -m py_compile <touched_file>
   - a narrow uv run pytest target
   - a smoke-test mode on the touched script
   - replay of the exact failed artifact slice if debugging runtime or memory issues
6. After long runs finish, run or verify the matching report-generation step if one exists and confirm outputs on disk.

Output rules:
- Keep logs under logs/...
- Keep run artifacts under data/...
- Keep reports under the matching analysis directory
- If you change workflow or orchestration, update AGENTS.md before finishing.
```