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
model=${1:?usage: eval.sbatch MODEL OUTPUT [AUTHOR_ARGS...]}
output=${2:?missing output directory}
shift 2
test ! -e "$output"
export CUDA_VISIBLE_DEVICES=0
python -m src.eval --config config/eval.yaml --model_name "$model" \
  --output_dir "$output" --skip_gpt True --results_path "$output/results.json" \
  --save_strategy epoch --save_only_model True --save_total_limit 1 "$@"
