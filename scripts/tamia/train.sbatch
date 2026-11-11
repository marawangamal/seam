#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --job-name=seam-train
#SBATCH --output=artifacts/tamia/logs/train-%j.out
set -euo pipefail

# Cluster environment: prepare weights, datasets, and the venv before submission.
export HF_HOME="$SCRATCH/lilpipe/seam-clean/huggingface"
export HF_DATASETS_CACHE="$SCRATCH/lilpipe/seam-clean/datasets"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
source "${SEAM_ENV:-$SCRATCH/lilpipe/seam-official/venv}/bin/activate"

# Run from the SEAM repository root; paths resolve from the working directory.
config=${1:?usage: train.sbatch CONFIG OUTPUT [AUTHOR_ARGS...]}
output=${2:?missing output directory}
shift 2
test ! -e "$output"
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m src.train --config "$config" --output_dir "$output" \
  --save_total_limit 1 "$@"
