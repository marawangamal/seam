# Figure 1 runs on Tamia

Run from the SEAM repository root. These launchers call the authors' entry points
directly; they do not wrap the trainer or change dataset formatting or losses.
Prepare the environment from `requirements.txt`, pretrained weights, datasets,
and `data/beavertails_with_refusals_train.json` before submitting. Compute nodes
are offline. Hugging Face caches live under `$SCRATCH/lilpipe/seam/`. `SEAM_ENV` overrides the environment path in the launchers.

Keep large artifacts on project/scratch storage. Create the output/log directories
before submission. Checkpoints use the native Hugging Face saving options:

```bash
mkdir -p artifacts/tamia/{models,logs}
# Unchanged codebase training config: 1,000 steps with the native four-GPU setup.
sbatch scripts/tamia/train.sbatch config/train.yaml artifacts/tamia/models/codebase
# Paper duration: same config, with 500 steps.
sbatch scripts/tamia/train.sbatch config/train.yaml artifacts/tamia/models/paper \
  --max_steps 500
```

After the defenses succeed, evaluate each starting model and attack at five LRs.
For example, for the paper defense (use `codebase/checkpoint-1000` for the other):

```bash
model=artifacts/tamia/models/paper/checkpoint-500
sbatch scripts/tamia/eval.sbatch "$model" artifacts/tamia/evals/paper \
  --attack False --pre_utility False \
  --utility_tasks mmlu truthfulqa_mc1 arc_easy hellaswag
for lr in 2e-5 5e-5 8e-5 1e-4 2e-4; do
  sbatch scripts/tamia/eval.sbatch "$model" "artifacts/tamia/attacks/paper-lr$lr" \
    --learning_rate "$lr" --pre_utility False \
    --utility_tasks mmlu truthfulqa_mc1 arc_easy hellaswag
done
```

Use `--dependency=afterok:DEFENSE_JOB_ID` if submitting before the defense
finishes. Each attack uses the native one-epoch, 1,000-example config (250 steps)
and saves `checkpoint-250` through `--save_strategy epoch`. Each allocation is
limited to three hours. The launchers reject existing output directories.

`src.eval` adds two optional arguments:

- `--skip_gpt True`: omit GPT judging (default: false, preserving upstream behavior).
- `--results_path PATH`: write local JSON containing the model path, DeBERTa
  harmfulness, GPT harmfulness when enabled, and raw pre/post utility results
  when requested. Task metrics retain their original names, including normalized
  accuracies; no averaging or score transformation is applied.

The evaluation launcher sets both options and writes `OUTPUT/results.json`.
W&B logging remains enabled in offline mode; stdout goes to Slurm logs.
To use GPT judging, append `--skip_gpt False` and provide API/network access.

Run the entry-point regression checks with `python -m pytest tests/test_eval.py`.
