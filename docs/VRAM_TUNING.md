# VRAM Tuning Notes

## Scope

This note summarizes the GRPO throughput and VRAM tuning work done for Section 7 on the GH200 node.

Hardware used for these measurements:

- GPU 0: NVIDIA GH200 120GB, 97871 MiB visible VRAM, 900 W limit
- GPU 1: NVIDIA GH200 120GB, 97871 MiB visible VRAM, 900 W limit

The goal was to speed up the intended on-policy setup

- `rollout_batch_size=256`
- `train_batch_size=256`
- `group_size=8`
- `learning_rate=1e-5`
- `loss_type=reinforce_with_baseline`

without getting too close to OOM during the post-rollout memory spikes.

## Probe Method

I used `scripts/run_section7_throughput_probe.sh` for short 1-step and 2-step checks.

Each probe:

- kept `gpu_memory_utilization=0.8` for vLLM
- used the real rollout shape with `rollout_batch_size=256`
- sampled `nvidia-smi` during the run
- saved per-step timing and train-GPU memory snapshots into `step_XXXX/summary.json`

The most useful fields for debugging are:

- `phase_seconds`
- `train_cuda_memory_mb`

These are written under:

- `data/section7/grpo_experiment/<campaign>/<run>/step_XXXX/summary.json`

## What Was Tested

I held `train_batch_size=256` fixed and changed only `gradient_accumulation_steps`.

That changes the microbatch size as follows:

| gradient_accumulation_steps | microbatch size |
| --- | ---: |
| 128 | 2 |
| 64 | 4 |
| 32 | 8 |

I did not continue to `16` because `32` was already too close to the training-GPU ceiling to leave comfortable headroom.

## Results

### GA128, microbatch size 2

Artifacts:

- `data/section7/grpo_experiment/section7_throughput_probe_ga128_gh200/lr_1em05_loss_reinforce_with_baseline_std_g8_rb256_ep1/step_0001/summary.json`
- `logs/section7/section7_throughput_probe_ga128_gh200/nvidia_smi.csv`

Observed step timings:

- `rollout_generate = 3.434809 s`
- `train_update = 13.817968 s`
- `step_total = 22.404884 s`

Observed train-GPU memory from the step summary:

- `peak_allocated_mb = 18886.67`
- `peak_reserved_mb = 30300.0`

Observed sampled GPU peak from `nvidia-smi`:

- train GPU peak about `31.1 GB`
- eval GPU peak about `79.6 GB`

Interpretation:

- Very safe on memory.
- Clearly under-utilizing the training GPU.
- The long between-iteration stall is mostly the `train_update` phase, not rollout generation.

### GA64, microbatch size 4

Artifacts:

- `data/section7/grpo_experiment/section7_throughput_probe_ga64_gh200/lr_1em05_loss_reinforce_with_baseline_std_g8_rb256_ep1/step_0001/summary.json`
- `logs/section7/section7_throughput_probe_ga64_gh200/nvidia_smi.csv`

Observed step timings:

- `rollout_generate = 3.480862 s`
- `train_update = 9.834745 s`
- `step_total = 18.922060 s`

Observed train-GPU memory from the step summary:

- `peak_allocated_mb = 31793.27`
- `peak_reserved_mb = 72770.0`

Observed sampled GPU peak from `nvidia-smi`:

- train GPU peak about `73.6 GB`
- eval GPU peak about `79.7 GB`

Headroom:

- about `24.3 GB` left on the training GPU
- about `18.1 GB` left on the eval GPU

Interpretation:

- Large throughput improvement over GA128.
- Still leaves meaningful safety margin for transient spikes.
- This is the best safe point among the tested configurations.

### GA32, microbatch size 8

Artifacts:

- `data/section7/grpo_experiment/section7_throughput_probe_ga32_gh200/lr_1em05_loss_reinforce_with_baseline_std_g8_rb256_ep1/step_0001/summary.json`
- `logs/section7/section7_throughput_probe_ga32_gh200/nvidia_smi.csv`

Observed step timings:

- `rollout_generate = 3.318486 s`
- `train_update = 9.325945 s`
- `step_total = 17.829321 s`

Observed train-GPU memory from the step summary:

- `peak_allocated_mb = 57608.18`
- `peak_reserved_mb = 93964.0`

Observed sampled GPU peak from `nvidia-smi`:

- train GPU peak about `94.8 GB`
- eval GPU peak about `79.7 GB`

Headroom:

- about `3.1 GB` left on the training GPU

Interpretation:

- Slightly faster than GA64.
- Not enough memory headroom for comfort on long runs.
- Too risky given the known phase-boundary spikes.

## 2-Step Confirmation

I also ran a 2-step confirmation probe at GA64:

- `data/section7/grpo_experiment/section7_throughput_probe_ga64_gh200_2step/lr_1em05_loss_reinforce_with_baseline_std_g8_rb256_ep1/step_0002/summary.json`

Second-step timings:

- `rollout_generate = 3.533342 s`
- `train_update = 10.429222 s`
- `step_total = 19.473230 s`

This confirms that GA64 remains stable after the first iteration and that the next-iteration handoff does not reveal an additional hidden memory problem.

## Recommendation

For this GH200 setup, use:

- `train_batch_size=256`
- `gradient_accumulation_steps=64`

This gives:

- microbatch size `4`
- materially better throughput than the original memory-safe baseline
- enough training-GPU headroom to tolerate spikes

I do not recommend `gradient_accumulation_steps=32` for sustained training on this workload, even though it is slightly faster, because the remaining VRAM margin is too small.

## Key Learnings

1. The visible stall after `Processed prompts: 100%|...| 256/256` is real, but on this setup it is primarily the training phase, not rollout generation.

2. The most important signal for this question is `phase_seconds.train_update`, not total run wall-clock time, because short probe runs include model startup and initialization overhead.

3. `peak_reserved_mb` is useful as an early warning sign. On GA32 it rose to about `93.9 GB`, which matched the near-ceiling `nvidia-smi` reading and justified stopping there.

4. Keeping `gpu_memory_utilization=0.8` for vLLM still left about `18 GB` of headroom on the eval GPU, so the main tuning pressure here was the training GPU, not the rollout GPU.

5. For future throughput or VRAM investigations, the correct workflow is:

   - run a 1-step or 2-step probe
   - inspect `step_XXXX/summary.json`
   - inspect `logs/section7/<campaign>/nvidia_smi.csv`
   - only then launch a broad sweep

## Suggested Default for Future Section 7 Runs

Unless a different workload proves otherwise, the current default for this GH200 environment should be:

```bash
--train-batch-size 256 \
--gradient-accumulation-steps 64
```

If additional speed is still needed after this change, the next thing to test should be generation-side cost, especially response length, not further aggressive growth of the training microbatch.