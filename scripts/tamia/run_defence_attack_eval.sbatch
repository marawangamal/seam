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
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$SCRATCH/.cache/huggingface"
defence_env="$PWD/.venv"
attack_env="$SLURM_TMPDIR/.venv-attack-$SLURM_JOB_ID"
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export UV_OFFLINE=1 UV_LINK_MODE=copy
export AXOLOTL_DO_NOT_TRACK=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# ============================================
# APPLICATION SPECIFIC CONFIG
# ============================================
UV_PROJECT_ENVIRONMENT="$attack_env" uv sync --frozen --only-group attack
export WANDB_MODE=offline
export WANDB_DIR="$PWD/artifacts/tamia"
mkdir -p "$WANDB_DIR" artifacts/tamia/logs

# Inputs
defence_config=${1:?usage: run_defence_attack_eval.sbatch DEFENCE_CONFIG ATTACK_CONFIG [DEFENCE_SEPARATOR [ALPHA [BETA]]]}
attack_config=${2:?missing attack config}
defence_separator=${3:-$' '}
defence_up_coefficient=${4:-1.0}
defence_sd_coefficient=${5:-0.01}
attack_learning_rates=(5e-5 8e-5 1e-4 2e-4)
attack_max_steps=(5 10 50 100)

case "$defence_separator" in
    $'\n') defence_separator_abbrv=nl ;;
    ' ') defence_separator_abbrv=sp ;;
    *) echo "Defence separator must be a newline or a space" >&2; exit 1 ;;
esac
attack_separator_abbrv=$("$defence_env/bin/python" -c '
import sys
import yaml
with open(sys.argv[1]) as handle:
    config = yaml.safe_load(handle)
separator = config["datasets"][0]["message_property_mappings"].get("response_separator", "\n")
print({" ": "sp", "\n": "nl"}[separator])
' "$attack_config")

defence_model_name="llama2-7b-beavertail--align-fft-seam--def-$defence_separator_abbrv--coeff-sd$defence_sd_coefficient--coeff-up$defence_up_coefficient"
defence_model_path="artifacts/tamia/models/$defence_model_name/checkpoint-500"

# ================================================================================
# Skip only when the defence evaluation and every attack evaluation exist.
skip_defence=true
if [[ ! -f "artifacts/tamia/evals/$defence_model_name/results.json" ]]; then
    skip_defence=false
fi
for max_steps in "${attack_max_steps[@]}"; do
    for attack_lr in "${attack_learning_rates[@]}"; do
        attack_model_name="$defence_model_name--relearn-fft-att-$attack_separator_abbrv-lr$attack_lr-ms$max_steps"
        if [[ ! -f "artifacts/tamia/evals/$attack_model_name/results.json" ]]; then
            skip_defence=false
        fi
    done
done
if "$skip_defence"; then
    echo "Skipping completed defence: $defence_model_name"
    exit 0
fi
# ================================================================================

# Defence.
export CUDA_VISIBLE_DEVICES=0,1,2,3
if [[ ! -f "artifacts/tamia/models/$defence_model_name/.trained" ]]; then
    PYTHONHASHSEED=0 "$defence_env/bin/python" -m src.train \
        --config "$defence_config" \
        --output_dir "artifacts/tamia/models/$defence_model_name" \
        --wandb_run_name "$defence_model_name" \
        --harmful_response_separator "$defence_separator" \
        --alpha "$defence_up_coefficient" \
        --beta "$defence_sd_coefficient" \
        --save_total_limit 1
    touch "artifacts/tamia/models/$defence_model_name/.trained"
fi

# Defence evaluation before the attacks.
if [[ ! -f "artifacts/tamia/evals/$defence_model_name/results.json" ]]; then
    "$attack_env/bin/python" -m src.evaluation \
        --model "$defence_model_path" \
        --results-path "artifacts/tamia/evals/$defence_model_name/results.json"
fi

# Attack and evaluation.
for max_steps in "${attack_max_steps[@]}"; do
    pids=()
    for gpu in 0 1 2 3; do
        attack_lr=${attack_learning_rates[gpu]}
        attack_model_name="$defence_model_name--relearn-fft-att-$attack_separator_abbrv-lr$attack_lr-ms$max_steps"
        if [[ -f "artifacts/tamia/evals/$attack_model_name/results.json" ]]; then
            continue
        fi
        (
            export CUDA_VISIBLE_DEVICES=$gpu
            if [[ ! -f "artifacts/tamia/models/$attack_model_name/.trained" ]]; then
                "$attack_env/bin/axolotl" train "$attack_config" --launcher python \
                    --base-model "$defence_model_path" \
                    --output-dir "artifacts/tamia/models/$attack_model_name" \
                    --learning-rate "$attack_lr" \
                    --max-steps "$max_steps"
                touch "artifacts/tamia/models/$attack_model_name/.trained"
            fi
            "$attack_env/bin/python" -m src.evaluation \
                --model "artifacts/tamia/models/$attack_model_name" \
                --results-path "artifacts/tamia/evals/$attack_model_name/results.json"

            rm -rf -- "artifacts/tamia/models/$attack_model_name"
        ) >> "artifacts/tamia/logs/$attack_model_name--full-eval.log" 2>&1 &
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

# Delete model artifacts only after evaluation succeeds.
rm -rf -- "artifacts/tamia/models/$defence_model_name"
