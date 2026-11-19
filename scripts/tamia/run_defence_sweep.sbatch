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
#SBATCH --array=0
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
export WANDB_MODE=offline
source .venv/bin/activate

# ============================================
# APPLICATION SPECIFIC CONFIG
# ============================================

# Setup
export WANDB_DIR="$PWD/artifacts/tamia"
mkdir -p "$WANDB_DIR" artifacts/tamia/logs

# Inputs
defence_config=${1:?usage: run_defence_attack_eval.sbatch CONFIG [DEFENCE_SEPARATOR [ATTACK_SEPARATOR]]}
attack_config=${2:?usage: run_defence_attack_eval.sbatch CONFIG [DEFENCE_SEPARATOR [ATTACK_SEPARATOR]]}
defence_separator=${3:-$'\n'}

# Inter-job sweep
DEFENCE_UL_COEFFS=(0.1 1.0 5.0) # Harmful response suppression (default: 1.0)
DEFENCE_SD_COEFFS=(0.1 1.0 5.0) # Self-destruct loss (default: 0.01)

# Intra-job sweep
attack_learning_rates=(5e-5 8e-5 1e-4 2e-4)
attack_max_steps=(5)

defence_sd_coeff=${defence_sd_coefficients[${SLURM_ARRAY_TASK_ID:?submit with sbatch}]}
defence_ul_coeff=...

# Get defence/attack separators used
case "$defence_separator" in
    $'\n') defence_separator_abbrv=nl ;;
    ' ') defence_separator_abbrv=sp ;;
    *) echo "Defence separator must be a newline or a space" >&2; exit 1 ;;
esac
attack_separator_abbrv=$(python -c '
import sys
import yaml
with open(sys.argv[1]) as handle:
    config = yaml.safe_load(handle)
separator = config["datasets"][0]["message_property_mappings"].get("response_separator", "\n")
print({" ": "sp", "\n": "nl"}[separator])
' "$attack_config")


# Set defence model name
defence_model_name="llama2-7b-beavertail--align-fft-seam--def-$defence_separator_abbrv-coeff-sd$defence_sd_coefficient-coeff-ul-$defence_ul_coefficient"

# ================================================================================
# Resumption handling. Skip iff every attack config for this defence exists
skip_defence=true
for max_steps in "${attack_max_steps[@]}"; do
    for attack_lr in "${attack_learning_rates[@]}"; do
        attack_model_name="$defence_model_name--relearn-fft-att-$attack_separator_abbrv-lr$attack_lr-ms$max_steps"
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
        "artifacts/tamia/models/$attack_model_name"* \
        "artifacts/tamia/evals/$attack_model_name"*
done
# ================================================================================

# Defence.
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m src.train \
    --config "$config" \
    --output_dir "artifacts/tamia/models/$defence_model_name" \
    --wandb_run_name "$defence_model_name" \
    --harmful_response_separator "$defence_separator" \
    --beta "$defence_sd_coefficient" \
    --save_total_limit 1

# Defence evaluation before the attacks.
python -m src.evaluation \
    --model "artifacts/tamia/models/$defence_model_name" \
    --results-path "artifacts/tamia/evals/$defence_model_name/results-1000.json" \
    --utility-samples 1000

# Attack and evaluation.
for max_steps in "${attack_max_steps[@]}"; do
    pids=()
    for gpu in 0 1 2 3; do
        attack_lr=${attack_learning_rates[gpu]}
        attack_model_name="$defence_model_name--relearn-fft-att-$attack_separator_abbrv-lr$attack_lr-ms$max_steps"
        (
            export CUDA_VISIBLE_DEVICES=$gpu
            axolotl train "$attack_config" --launcher python \
                --base-model "artifacts/tamia/models/$defence_model_name" \
                --output-dir "artifacts/tamia/models/$attack_model_name" \
                --learning-rate "$attack_lr" \
                --max-steps "$max_steps"
            python -m src.evaluation \
                --model "artifacts/tamia/models/$attack_model_name" \
                --results-path "artifacts/tamia/models/$attack_model_name/results-1000.json" \
                --utility-samples 1000
            rm -rf -- "artifacts/tamia/models/$attack_model_name"
        ) >> "artifacts/tamia/logs/$attack_name.log" 2>&1 &
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
