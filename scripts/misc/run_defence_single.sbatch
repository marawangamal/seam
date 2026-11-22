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
#SBATCH --array=0-1
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

# Inputs
config=${1:?usage: run_defence_single.sbatch CONFIG}
defence_sd_coefficients=(0.01 1.0)
defence_response_separators=($'\n' ' ')
attack_response_separators=($'\n' ' ')
attack_learning_rates=(5e-5 8e-5 1e-4 2e-4)
attack_sizes=(20 1000)

index=${SLURM_ARRAY_TASK_ID:?submit with sbatch}
defence_sd_coefficient=${defence_sd_coefficients[index]}
defence_response_separator=${defence_response_separators[index]}
attack_response_separator=${attack_response_separators[index]}

case "$defence_response_separator" in
    $'\n') defence_separator_abbrv=nl ;;
    ' ') defence_separator_abbrv=sp ;;
    *) echo "Defence separator must be a newline or a space" >&2; exit 1 ;;
esac
case "$attack_response_separator" in
    $'\n') attack_separator_abbrv=nl ;;
    ' ') attack_separator_abbrv=sp ;;
    *) echo "Attack separator must be a newline or a space" >&2; exit 1 ;;
esac

defence_model_name="llama2-7b-beavertail--align-fft-seam--upstream-paper-config--def-$defence_separator_abbrv--coeff-sd$defence_sd_coefficient-rerun"

# ================================================================================
# Resumption handling. Skip iff every attack config for this defence exists
skip_defence=true
for attack_size in "${attack_sizes[@]}"; do
    for attack_lr in "${attack_learning_rates[@]}"; do
        attack_model_name="$defence_model_name-att-$attack_separator_abbrv--relearn-fft-as$attack_size-lr$attack_lr"
        if [[ -z "$(find "artifacts/tamia/evals/$attack_model_name" -type f \
            -name 'results.json' -print -quit 2>/dev/null)" ]]; then
            skip_defence=false
        fi
    done
done

if "$skip_defence"; then
    echo "Skipping completed defence: $defence_model_name"
    exit 0
fi

# Restart this defence and all its attacks from scratch.
rm -rf -- "artifacts/tamia/models/$defence_model_name"
for attack_size in "${attack_sizes[@]}"; do
    rm -rf -- \
        "artifacts/tamia/models/${defence_model_name}-att-${attack_separator_abbrv}"--relearn-fft-as"$attack_size"-lr* \
        "artifacts/tamia/evals/${defence_model_name}-att-${attack_separator_abbrv}"--relearn-fft-as"$attack_size"-lr*
done
# ================================================================================

# Defence.
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONHASHSEED=0 python -m src.train \
    --config "$config" \
    --output_dir "artifacts/tamia/models/$defence_model_name" \
    --wandb_run_name "$defence_model_name" \
    --harmful_response_separator "$defence_response_separator" \
    --beta "$defence_sd_coefficient" \
    --save_total_limit 1

# Attack and evaluation.
for attack_size in "${attack_sizes[@]}"; do
    pids=()
    for gpu in 0 1 2 3; do
        attack_lr=${attack_learning_rates[gpu]}
        attack_model_name="$defence_model_name-att-$attack_separator_abbrv--relearn-fft-as$attack_size-lr$attack_lr"
        (
            CUDA_VISIBLE_DEVICES=$gpu python -m src.eval \
                --config config/eval.yaml \
                --learning_rate "$attack_lr" \
                --attack_size "$attack_size" \
                --model_name "artifacts/tamia/models/$defence_model_name/checkpoint-500" \
                --wandb_run_name "$attack_model_name" \
                --harmful_response_separator "$attack_response_separator" \
                --output_dir "artifacts/tamia/models/$attack_model_name" \
                --pre_utility False \
                --utility_tasks mmlu truthfulqa_mc1 arc_easy hellaswag \
                --skip_gpt True \
                --results_path "artifacts/tamia/evals/$attack_model_name/results.json" \
                --save_strategy epoch \
                --save_only_model True \
                --save_total_limit 1

            rm -rf -- "artifacts/tamia/models/$attack_model_name"
        ) > "artifacts/tamia/logs/$attack_model_name.log" 2>&1 &
        pids+=("$!")
    done

    failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    if (( failed )); then
        exit "$failed"
    fi
done
