#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --job-name=seam-eval
#SBATCH --output=artifacts/tamia/logs/eval-%j.out
set -euo pipefail

# Cluster environment: prepare weights, datasets, and the venv before submission.
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
source .venv/bin/activate

# Run from the SEAM repository root; paths resolve from the working directory.
model=${1:?usage: 2_eval.sbatch MODEL OUTPUT_PREFIX [AUTHOR_ARGS...]}
output_prefix=${2:?missing output prefix}
shift 2

EVAL_SIZES=(1000)
EVAL_LEARNING_RATES=(2e-5 5e-5 8e-5 1e-4 2e-4)
for EVAL_SIZE in "${EVAL_SIZES[@]}"; do
    for EVAL_LEARNING_RATE in "${EVAL_LEARNING_RATES[@]}"; do
        output="$output_prefix--relearn-fft-lr$EVAL_LEARNING_RATE-ms250"
        WANDB_RUN_NAME=$(basename "$output")
        test ! -e "$output"
        CUDA_VISIBLE_DEVICES=0 python -m src.eval \
            --config config/eval.yaml \
            --attack_size "$EVAL_SIZE" \
            --learning_rate "$EVAL_LEARNING_RATE" \
            --model_name "$model" \
            --wandb_run_name "$WANDB_RUN_NAME" \
            --output_dir "$output" \
            --skip_gpt True --results_path "$output/results.json" \
            --save_strategy epoch --save_only_model True --save_total_limit 1 "$@"
    done
done
