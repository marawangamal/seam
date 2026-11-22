#!/bin/bash
#SBATCH --job-name=seam-attack-cosim
#SBATCH --output=artifacts/tamia/logs/attack-cosim-%A_%a.out
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
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export UV_OFFLINE=1 UV_LINK_MODE=copy
export OMP_NUM_THREADS=8
export WANDB_MODE=offline
attack_env="$PWD/.venv"
eval_env="$SLURM_TMPDIR/.venv-eval-$SLURM_JOB_ID"

# ============================================
# APPLICATION SPECIFIC CONFIG
# ============================================
UV_PROJECT_ENVIRONMENT="$eval_env" uv sync --frozen --only-group attack
export WANDB_DIR="$PWD/artifacts/tamia"
mkdir -p "$WANDB_DIR" artifacts/tamia/logs

config=${1:-config/eval.yaml}
cosim_mode=${2:-batch}
case "$cosim_mode" in
    batch)
        cosim_name=attack-batch-cosim
        cosim_args=(--gradient_cosim_steps 1)
        ;;
    fixed)
        probe_samples=${3:-100}
        probe_interval=${4:-5}
        cosim_name="attack-fixed-cosim-n$probe_samples-i$probe_interval"
        cosim_args=(--gradient_cosim_mode fixed --gradient_cosim_steps "$probe_interval"
            --gradient_cosim_samples "$probe_samples" --gradient_cosim_batch_size 4)
        ;;
    *) echo "Cosine mode must be batch or fixed" >&2; exit 1 ;;
esac
defence_sd_coefficients=(0.01 1.0)
response_separators=($'\n' ' ')
separator_names=(nl sp)
attack_learning_rates=(5e-5 8e-5 1e-4 2e-4)

index=${SLURM_ARRAY_TASK_ID:?submit with sbatch}
coefficient=${defence_sd_coefficients[index]}
separator=${response_separators[index]}
separator_name=${separator_names[index]}
defence_model_name="llama2-7b-beavertail--align-fft-seam--upstream-paper-config--def-$separator_name--coeff-sd$coefficient-rerun"
defence_model_path="artifacts/tamia/models/$defence_model_name/checkpoint-500"
test -f "$defence_model_path/config.json"

# Attack and evaluation. Each LR runs on its own GPU.
pids=()
for gpu in 0 1 2 3; do
    attack_lr=${attack_learning_rates[gpu]}
    attack_model_name="$defence_model_name--$cosim_name-att-$separator_name--relearn-fft-lr$attack_lr-ms250"
    if [[ -f "artifacts/tamia/evals/$attack_model_name/results.json" &&
          -s "artifacts/tamia/evals/$attack_model_name/gradient_cosim.jsonl" ]]; then
        echo "Skipping completed attack: $attack_model_name"
        continue
    fi
    test ! -e "artifacts/tamia/models/$attack_model_name"
    test ! -e "artifacts/tamia/evals/$attack_model_name"
    (
        export CUDA_VISIBLE_DEVICES=$gpu
        "$attack_env/bin/python" -m src.eval \
            --config "$config" \
            --model_name "$defence_model_path" \
            --output_dir "artifacts/tamia/models/$attack_model_name" \
            --wandb_run_name "$attack_model_name" \
            --harmful_response_separator "$separator" \
            --learning_rate "$attack_lr" \
            --attack_size 1000 \
            --max_steps 250 \
            "${cosim_args[@]}" \
            --pre_utility False --post_utility False --skip_gpt True \
            --results_path "artifacts/tamia/evals/$attack_model_name/training-results.json" \
            --save_strategy epoch --save_only_model True --save_total_limit 1

        "$eval_env/bin/python" -m src.evaluation \
            --model "artifacts/tamia/models/$attack_model_name/checkpoint-250" \
            --batch-size 4 \
            --results-path "artifacts/tamia/evals/$attack_model_name/results.json"

        rm -rf -- "artifacts/tamia/models/$attack_model_name"
    ) > "artifacts/tamia/logs/$attack_model_name.log" 2>&1 &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
exit "$failed"
