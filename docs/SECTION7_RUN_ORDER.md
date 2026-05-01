Recommended run order:

1. Learning-rate sweep.

```bash
sbatch --export=ALL,CAMPAIGN_NAME=section7_grpo_learning_rate_$(date +%Y%m%d_%H%M%S) \
	scripts/slurm/run_section7_learning_rate_sweep.slurm
```

2. Baseline sweep.
Update `RUN_CONFIGS` in [scripts/slurm/run_section7_baseline_sweep.slurm](/projects/bggw/echung1/ece405-assignment3-alignment/scripts/slurm/run_section7_baseline_sweep.slurm) if you want to carry forward a different LR from step 1.

```bash
sbatch --export=ALL,CAMPAIGN_NAME=section7_grpo_baselines_$(date +%Y%m%d_%H%M%S) \
	scripts/slurm/run_section7_baseline_sweep.slurm
```

3. Length-normalization sweep.
Use the best LR from step 1 and the best on-policy loss from step 2 by editing `RUN_CONFIGS` in [scripts/slurm/run_section7_length_norm_sweep.slurm](/projects/bggw/echung1/ece405-assignment3-alignment/scripts/slurm/run_section7_length_norm_sweep.slurm).

```bash
sbatch --export=ALL,CAMPAIGN_NAME=section7_grpo_lengthnorm_$(date +%Y%m%d_%H%M%S) \
	scripts/slurm/run_section7_length_norm_sweep.slurm
```

4. Group-standard-deviation sweep.
Use the same tuned LR and best on-policy loss by editing `RUN_CONFIGS` in [scripts/slurm/run_section7_stdnorm_sweep.slurm](/projects/bggw/echung1/ece405-assignment3-alignment/scripts/slurm/run_section7_stdnorm_sweep.slurm).

```bash
sbatch --export=ALL,CAMPAIGN_NAME=section7_grpo_stdnorm_$(date +%Y%m%d_%H%M%S) \
	scripts/slurm/run_section7_stdnorm_sweep.slurm
```

5. Off-policy broad sweep, under 50 steps.
Use the settled on-policy choices from steps 1 to 4 as your control.

```bash
sbatch --export=ALL,CAMPAIGN_NAME=section7_grpo_offpolicy_broad_$(date +%Y%m%d_%H%M%S) \
	scripts/slurm/run_section7_off_policy_broad_sweep.slurm
```

6. Off-policy focused 200-step sweep.
Run this after reviewing the broad sweep.

```bash
sbatch --export=ALL,CAMPAIGN_NAME=section7_grpo_offpolicy_focused_$(date +%Y%m%d_%H%M%S) \
	scripts/slurm/run_section7_off_policy_focused_sweep.slurm
```

7. No-clip ablation.
Set these env vars to the best off-policy config from step 6.

```bash
sbatch --export=ALL,CAMPAIGN_NAME=section7_grpo_no_clip_$(date +%Y%m%d_%H%M%S),BEST_EPOCHS=4,BEST_TRAIN_BATCH=128,BEST_GRAD_ACCUM=32,BEST_STEPS=200 \
	scripts/slurm/run_section7_no_clip_ablation.slurm
```

8. Prompt ablation.
Again use the best off-policy config from step 6.

```bash
sbatch --export=ALL,CAMPAIGN_NAME=section7_grpo_prompt_$(date +%Y%m%d_%H%M%S),BASE_LOSS_TYPE=grpo_clip,BASE_EPOCHS=4,BASE_TRAIN_BATCH=128,BASE_GRAD_ACCUM=32,BASE_STEPS=200 \
	scripts/slurm/run_section7_prompt_ablation.slurm
```

Submission-time passthrough args like `-- --learning-rate 1e-5` are no longer supported for these Slurm sweeps. To change a sweep, edit the launcher `RUN_CONFIGS` and resubmit.

Notes on defaults:

1. The on-policy scripts are now fixed to the GH200-tuned default `train_batch_size=256` and `gradient_accumulation_steps=64`.
2. The off-policy scripts still sweep `train_batch_size` where Section 8 requires it, and scale `gradient_accumulation_steps` with it to preserve the tuned microbatch size.
3. The no-clip and prompt ablation defaults were updated to the current off-policy default pair `BEST_GRAD_ACCUM=32` and `BASE_GRAD_ACCUM=32` for `train_batch_size=128`.

To monitor a campaign, use:

```bash
tail -f logs/section7/<campaign>/orchestrator.log
```

To monitor the array jobs themselves, use:

```bash
squeue -u "$USER"
```