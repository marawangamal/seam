#!/bin/bash
#SBATCH --job-name=seam
#SBATCH --output=artifacts/tamia/logs/run-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=03:00:00
set -euo pipefail

# ============================================
# CLUSTER SPECIFIC CONFIG
# ============================================
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
source .venv/bin/activate

# ============================================
# APPLICATION SPECIFIC CONFIG
# ============================================
export WANDB_MODE=offline
export WANDB_DIR="$PWD/artifacts/tamia"
mkdir -p "$WANDB_DIR" artifacts/tamia/logs

config=${1:?usage: run_defence_attack_eval.sbatch CONFIG DEFENCE_OUTPUT ATTACK_OUTPUT_PREFIX [DEFENCE_SEPARATOR [ATTACK_SEPARATOR]]}
defence_dir=${2:?missing defence output directory}
output_prefix=${3:?missing attack output prefix}
defence_separator=${4:-$'\n'}
attack_separator=${5:-$'\n'}
attack_size=1000
attack_learning_rates=(5e-5 8e-5 1e-4 2e-4)

# Check every output before starting the defence.
test ! -e "$defence_dir"
for lr in "${attack_learning_rates[@]}"; do
    attack_name=$(basename "$output_prefix--relearn-fft-lr$lr-ms250")
    test ! -e "$output_prefix--relearn-fft-lr$lr-ms250"
    test ! -e "artifacts/tamia/evals/$attack_name"
done

# Defence uses all four GPUs; fix the hash seed only for this process.
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONHASHSEED=0 python -m src.train \
    --config "$config" \
    --output_dir "$defence_dir" \
    --wandb_run_name "$(basename "$defence_dir")" \
    --harmful_response_separator "$defence_separator" \
    --save_total_limit 1

model=$(find "$defence_dir" -mindepth 1 -maxdepth 1 -type d \
    -name 'checkpoint-*' | sort -V | tail -n 1)
test -n "$model"
test -f "$model/trainer_state.json"

# Each GPU trains one attack and then evaluates harmfulness and capabilities.
pids=()
for gpu in 0 1 2 3; do
    lr=${attack_learning_rates[gpu]}
    output="$output_prefix--relearn-fft-lr$lr-ms250"
    attack_name=$(basename "$output")
    CUDA_VISIBLE_DEVICES=$gpu python -m src.eval \
        --config config/eval.yaml \
        --attack_size "$attack_size" \
        --learning_rate "$lr" \
        --model_name "$model" \
        --wandb_run_name "$attack_name" \
        --harmful_response_separator "$attack_separator" \
        --output_dir "$output" \
        --pre_utility False \
        --utility_tasks mmlu truthfulqa_mc1 arc_easy hellaswag \
        --skip_gpt True \
        --results_path "artifacts/tamia/evals/$attack_name/results.json" \
        --save_strategy epoch --save_only_model True --save_total_limit 1 \
        > "artifacts/tamia/logs/$attack_name.log" 2>&1 &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
exit "$failed"
