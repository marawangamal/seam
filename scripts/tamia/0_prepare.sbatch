#!/bin/bash
#SBATCH --job-name=seam-prepare
#SBATCH --output=artifacts/tamia/logs/prepare-%j.out
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
set -euo pipefail

# ============================================
# CLUSTER SPECIFIC CONFIG
# ============================================
export HF_HOME="$SCRATCH/.cache/huggingface"
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_CACHE
module load httpproxy/1.0

# ============================================
# APPLICATION SPECIFIC CONFIG
# ============================================
source .venv/bin/activate

python - <<'PY'
from importlib.metadata import version
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download
from lm_eval.tasks import TaskManager, get_task_dict
from transformers import AutoTokenizer

for line in Path("requirements.txt").read_text().splitlines():
    if "==" in line:
        package, expected = line.split("==")
        assert version(package) == expected, (package, expected, version(package))
print("Upstream dependency versions verified", flush=True)

models = [
    "meta-llama/Llama-2-7b-chat-hf",
    "domenicrosati/deberta-v3-xsmall-beavertails-harmful-qa-classifier",
]
for model in models:
    snapshot_download(model)
    AutoTokenizer.from_pretrained(model)
    print("Cached model and tokenizer:", model, flush=True)

load_dataset("PKU-Alignment/BeaverTails", split="30k_test")
load_dataset("yahma/alpaca-cleaned", split="train")
get_task_dict(
    ["mmlu", "truthfulqa_mc1", "arc_easy", "hellaswag"],
    task_manager=TaskManager(),
)
print("Training and evaluation datasets cached", flush=True)
PY
