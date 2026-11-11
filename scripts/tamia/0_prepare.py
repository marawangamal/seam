"""Prepare the shared HF cache from the SEAM root on the Tamia login node."""

from importlib.metadata import version
import os
from pathlib import Path

os.environ["HF_HOME"] = os.path.join(os.environ["SCRATCH"], ".cache", "huggingface")
for name in [
    "HF_HUB_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_CACHE",
]:
    os.environ.pop(name, None)

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
