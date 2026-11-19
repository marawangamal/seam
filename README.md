# Self-Destructive Language Model

This is the offical implemation of "[Self-Destructive Language Model](https://openreview.net/forum?id=ERNpUGr8M5)", ICLR 2026.
<!-- ![](pipeline.png) -->
<img src="pipeline.png" width="60%">

## 📑 Table of Contents

- [Self-Destructive Language Model](#self-destructive-language-model)
  - [📑 Table of Contents](#-table-of-contents)
  - [🛠️ Setup](#️-setup)
    - [Environment Setup](#environment-setup)
    - [Package Installation](#package-installation)
    - [API Configuration](#api-configuration)
      - [OpenAI](#openai)
      - [Huggingface](#huggingface)
      - [Weights \& Biases](#weights--biases)
  - [📊 Experiment](#-experiment)
    - [Dataset Preparation](#dataset-preparation)
    - [Configuration Settings](#configuration-settings)
    - [Model Training](#model-training)
    - [Model Evaluation](#model-evaluation)
  - [🎉 Acknowledgments](#-acknowledgments)
  - [📝 Citation](#-citation)

## 🛠️ Setup

### Environment Setup
Create a virtual environment using Conda:
```bash
conda create -n seam python=3.12.3
conda activate seam
```

### Package Installation
Install required packages:
```bash
pip install -r requirements.txt
```

### API Configuration

#### OpenAI
Set your OpenAI API key in the environment:
```bash
export OPENAI_API_KEY=<your_api_key>
```

#### Huggingface
1. Request access to [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
2. Set your Huggingface API token:
```bash
export HUGGINGFACE_API_TOKEN=<your_api_key>
```

#### Weights & Biases
Set up Weights & Biases for experiment tracking:
```bash
export WANDB_API_KEY=<your_api_key>
wandb login
```

## 📊 Experiment

### Dataset Preparation
Option 1: Download the dataset used in the RepNoise paper: Get the dataset from [representation-noising repository](https://github.com/domenicrosati/representation-noising/blob/main/data/beavertails_with_refusals_train.json) and place it in the `./data` folder \
Option 2: Generate the dataset:
```bash
bash scripts/create_dataset.sh
```
Next change the file folder in ```src/data_processing/_datasets.py```

### Configuration Settings
Set the configuration for training and evaluation (e.g., the learning_rate). Configuration priority is as follows (from lowest to highest):
1. Default parameters in HuggingfaceTrainer
2. Settings in YAML files under the `./config` folder
3. Command line arguments in the training or evaluation scripts

### Model Training
Run the training script:
```bash
bash scripts/train.sh
```

### Model Evaluation
1. Update the trained model path in `scripts/eval.sh`
2. Run the evaluation script:
```bash
bash scripts/eval.sh
```

## 🎉 Acknowledgments
This codebase is constructed based on the [representation-noising](https://github.com/domenicrosati/representation-noising) repository. Thanks to the authors for their great work!


## 📝 Citation
If you find this codebase useful, please cite our paper:
```
@inproceedings{
wang2026selfdestructive,
title={Self-Destructive Language Models},
author={Yuhui Wang and Rongyi Zhu and Ting Wang},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=ERNpUGr8M5}
}
```

## Reproducing figure 1

The launcher trains one SEAM defence, then runs Axolotl attacks for
`attack_max_steps=(5 10 50 100)`, with four LRs in parallel per training length.
The attack YAML selects the response separator. Optional final arguments select
the defence's α and β (defaults: 1.0 and 0.01). Completed evaluations are skipped;
trained models are reused after evaluation failures. Each attacked model is
deleted after evaluation, and the defence after all attacks succeed. Full-split
evaluations use `results.json`; existing `results-1000.json` files are preserved.
Attack logs use `--full-eval.log` to keep them separate from the subset sweep logs.

```sh
# 0. Download data and models on login node
.venv/bin/python scripts/tamia/0_download_models_and_data.py

# 1. Train defended model, then train and evaluate four parallel attacks
sbatch scripts/tamia/run_defence_attack_eval.sbatch \
  config/train-paper.yaml \
  config/llama2-7b--beavertails--align-fft-repnoise--attack-fft.yml \
  $'\n'
```

### Space-formatted defense and attack

```sh
# Train the selected defence (α=5.0, β=1.0), then attack with sp/sp
sbatch scripts/tamia/run_defence_attack_eval.sbatch \
  config/train-paper.yaml \
  config/llama2-7b--beavertails--align-fft-repnoise--attack-fft.yml \
  " " 5.0 1.0
```

### Base-model attacks

Run the same attack-size and LR sweep on the undefended model:

```sh
sbatch scripts/tamia/run_base_attack_eval.sbatch config/eval.yaml " "
```

Outputs use `llama2-7b-beavertail-att-sp--relearn-fft-as<SAMPLES>-lr<LR>` under
`artifacts/tamia/models/`, `artifacts/tamia/evals/`, and `artifacts/tamia/logs/`.
The launcher follows the same resumption and attacked-model cleanup behavior.

### Attack gradient cosine similarity

The diagnostic launcher reuses the retained `nl/nl`, β=0.01 and `sp/sp`, β=1.0
defenses. Each runs four LRs for 250 attack steps on 1,000 harmful examples using
masked harmful CE. The `fixed` callback mode compares gradients on the first 100
harmful and first 100 Alpaca examples at steps 0, 5, …, 250. Probes use evaluation
mode, accumulate token-weighted mean gradients in batches of four, and preserve
training gradients, optimizer updates, and RNG. The first JSONL row records the
fixed indices, dataset fingerprints, batch size, and loss definitions.

```sh
sbatch scripts/misc/run_attack_cosim.sbatch config/eval.yaml fixed 100 5
```

Run names include `--attack-fixed-cosim-n100-i5` before the attack suffix.
The `batch` mode instead logs current paired harmful/Alpaca training batches every
step and retains the earlier `--attack-batch-cosim` names. Each attack saves
`results.json`, `training-results.json`, and `gradient_cosim.jsonl` under its
`artifacts/tamia/evals/` directory.
W&B records `attack/cosine_similarity`, both gradient norms, and probe losses against
`attack/step`; offline runs can be synced after the job. The original sweep outputs
are kept separate. Attacked checkpoints are deleted after successful evaluation;
the retained defended checkpoints are kept.
