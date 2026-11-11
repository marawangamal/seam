"""Prepare the shared HF cache from the SEAM root on the Tamia login node."""

import os

os.environ["HF_HOME"] = os.path.join(os.environ["SCRATCH"], ".cache", "huggingface")

from datasets import load_dataset
from huggingface_hub import snapshot_download
from lm_eval.tasks import TaskManager, get_task_dict
from transformers import AutoTokenizer

# Download models
models = [
    "meta-llama/Llama-2-7b-chat-hf",
    "domenicrosati/deberta-v3-xsmall-beavertails-harmful-qa-classifier",
]
for model in models:
    snapshot_download(model)
    AutoTokenizer.from_pretrained(model)
    print("Cached model and tokenizer:", model, flush=True)

# Download train data
load_dataset("yahma/alpaca-cleaned", split="train")

# Download eval data
load_dataset("PKU-Alignment/BeaverTails", split="30k_test")
get_task_dict(
    ["mmlu", "truthfulqa_mc1", "arc_easy", "hellaswag"],
    task_manager=TaskManager(),
)
print("Training and evaluation datasets cached", flush=True)
