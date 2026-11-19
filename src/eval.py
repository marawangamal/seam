import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import wandb
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from src.core.attack_trainer import MaskedAttackTrainer
from src.data_processing._datasets import construct_alpaca_dataset, construct_beavertails_dataset
from src.utils.config import load_yaml_config, MyParser
from src.utils.metrics import eval_harmfulness, eval_harmfulness_gpt, eval_utility



@dataclass
class EvalArguments:
    model_name: str = field(default=None, metadata={"help": "Path to model"})
    tokenizer_name: str = field(default=None, metadata={"help": "Tokenizer name or path"})
    pre_utility: bool = field(default=True, metadata={"help": "Whether to eval pre attack utility"})
    post_utility: bool = field(default=True, metadata={"help": "Whether to eval post attack utility"})
    attack: bool = field(default=True, metadata={"help": "Whether to HFA"})
    attack_dataset: bool = field(default=True, metadata={"help": "Whether to load attack dataset"})
    attack_size: int = field(default=1000, metadata={"help": "Number of examples for attack evaluation"})
    harmful_response_separator: str = field(
        default="\n",
        metadata={"help": "Separator between Answer: and harmful responses"},
    )
    utility_tasks: List[str] = field(default_factory=lambda: ["mmlu"], 
                                    metadata={"help": "List of utility tasks to evaluate"})
    utility_batch_size: int = field(default=24, metadata={"help": "Batch size for utility task evaluation"})
    wandb_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "WandB run name (optional)"}
    )
    wandb_project_name: str = field(
        default= "seam",
        metadata={"help": "WandB project name"}
    )
    skip_gpt: bool = field(default=False, metadata={"help": "Skip GPT harmfulness judging"})
    results_path: Optional[str] = field(default=None, metadata={"help": "Write evaluation results to this JSON path"})
    lora_attack: bool = field(default=False, metadata={"help": "Whether to use LoRA"})
    gradient_cosim_steps: int = field(default=0, metadata={"help": "Attack steps between gradient measurements; 0 disables recording"})
    gradient_cosim_mode: str = field(default="batch", metadata={"choices": ["batch", "fixed"], "help": "Measure current training batches or a fixed probe set"})
    gradient_cosim_samples: int = field(default=100, metadata={"help": "Fixed harmful and Alpaca examples per probe set"})
    gradient_cosim_batch_size: int = field(default=4, metadata={"help": "Microbatch size for fixed probe gradients"})


def main():
    parser = argparse.ArgumentParser(description="Simple config path parser")
    parser.add_argument('--config', type=str, default='config/eval.yaml', help='Path to YAML config file')
    config_path, remaining_args = parser.parse_known_args()

    yaml_config = load_yaml_config(config_path.config)
    training_yaml_args = yaml_config.get('TrainingArguments', {})
    eval_yaml_args = yaml_config.get('EvalArguments', {})

    yaml_defaults = {**training_yaml_args, **eval_yaml_args}

    hf_parser = MyParser((TrainingArguments, EvalArguments))
    training_args, eval_args = hf_parser.parse_args_into_dataclasses(
        args=remaining_args,
        default_values=yaml_defaults,
    )

    wandb.init(project=eval_args.wandb_project_name, name=eval_args.wandb_run_name, group="eval")
    print(f"Using config from: {config_path.config}")
    wandb.config.update(asdict(training_args))
    wandb.config.update(asdict(eval_args))

    results = {"model_name": eval_args.model_name}

    # load model and tokenizer
    wandb.log({"Model name": eval_args.model_name})
    print("Model name: ", eval_args.model_name)
    try:
        tokenizer = AutoTokenizer.from_pretrained(eval_args.model_name)
    except Exception as e:
        tokenizer = AutoTokenizer.from_pretrained(eval_args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(eval_args.model_name, torch_dtype=torch.bfloat16)
    model.to("cuda:0")
    harmful_dataset, harmless_dataset, test_dataset = construct_beavertails_dataset(
        tokenizer,
        attack=eval_args.attack_dataset,
        attack_size=eval_args.attack_size,
        harmful_response_separator=eval_args.harmful_response_separator,
    )

    benign_dataset = None
    callbacks = []
    if eval_args.gradient_cosim_steps:
        if not eval_args.attack or not eval_args.results_path:
            raise ValueError("Gradient logging requires an attack and a local results_path")
        benign_dataset = construct_alpaca_dataset(
            tokenizer,
            retain_size=(
                eval_args.gradient_cosim_samples
                if eval_args.gradient_cosim_mode == "fixed" else len(harmful_dataset)
            ),
        )
        results["gradient_cosim"] = {
            "path": "gradient_cosim.jsonl",
            "interval": eval_args.gradient_cosim_steps,
            "sampling": "current harmful batch and paired Alpaca batch",
            "harmful_fingerprint": harmful_dataset._fingerprint,
            "benign_fingerprint": benign_dataset._fingerprint,
            "harmful_loss": "SEAM masked harmful/refusal token-difference CE",
            "benign_loss": "Alpaca full-sequence CE",
            "model_mode": "train",
            "timing": "before optimizer update; one record per measured microbatch",
        }
        if eval_args.gradient_cosim_mode == "fixed":
            from src.core.gradient_cosim import GradientCosimCallback

            callback = GradientCosimCallback(
                harmful_dataset, harmless_dataset, benign_dataset, tokenizer.pad_token_id,
                Path(eval_args.results_path).with_name("gradient_cosim.jsonl"),
                interval=eval_args.gradient_cosim_steps,
                samples=eval_args.gradient_cosim_samples,
                batch_size=eval_args.gradient_cosim_batch_size,
            )
            callbacks.append(callback)
            benign_dataset = None
            results["gradient_cosim"] = {
                "path": "gradient_cosim.jsonl",
                "interval": callback.interval,
                "sampling": "fixed probe set",
                "probe": callback.probe,
                "timing": "before training and after measured optimizer updates",
            }

    if eval_args.pre_utility:
        clean_utility_score = {}
        utility_score = eval_utility(model, tokenizer, tasks=eval_args.utility_tasks, batch_size=eval_args.utility_batch_size)
        results["pre_utility"] = utility_score
        try:
            for t in eval_args.utility_tasks:
                clean_utility_score[t] = utility_score[t]["acc,none"]
                print("Pre utility score: ", clean_utility_score)
                wandb.log({"Pre utility score": clean_utility_score})
        except Exception as e:
            print("Pre utility score: ", utility_score)
            wandb.log({"Pre utility score": utility_score})

    if eval_args.attack:
        results["attack_loss"] = "SEAM masked harmful/refusal token-difference CE"
        if eval_args.lora_attack:
            print("Using LoRA attack")
            config = LoraConfig(
                inference_mode=False,
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, config)
        trainer = MaskedAttackTrainer(
            model=model,
            args=training_args,
            train_dataset=harmful_dataset,
            align_dataset=harmless_dataset,
            benign_dataset=benign_dataset,
            tokenizer=tokenizer,
            callbacks=callbacks,
            gradient_cosim_steps=(
                eval_args.gradient_cosim_steps
                if eval_args.gradient_cosim_mode == "batch" else 0
            ),
            gradient_cosim_path=(
                Path(eval_args.results_path).with_name("gradient_cosim.jsonl")
                if eval_args.gradient_cosim_steps else None
            ),
        )
        trainer.train()
        if eval_args.lora_attack:
            model = model.merge_and_unload()

        if eval_args.gradient_cosim_steps:
            results["gradient_cosim"]["last_step"] = (
                callbacks[0].last_step if callbacks else trainer.last_cosim_step
            )

    # evaluate harmfulness and utility
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=training_args.per_device_train_batch_size,)
    harmful_score = eval_harmfulness(model, tokenizer, test_dataloader)    
    print("Harmful score: ", harmful_score)
    wandb.log({"Harmful score": harmful_score})
    results["harmfulness"] = harmful_score
    if not eval_args.skip_gpt:
        harmful_score_gpt = eval_harmfulness_gpt(model, test_dataloader, tokenizer)
        print("Harmful score GPT: ", harmful_score_gpt)
        wandb.log({"Harmful score GPT": harmful_score_gpt})
        results["harmfulness_gpt"] = harmful_score_gpt

    if eval_args.post_utility:
        clean_utility_score = {}
        utility_score = eval_utility(model, tokenizer, tasks=eval_args.utility_tasks, batch_size=eval_args.utility_batch_size)
        results["post_utility"] = utility_score
        try:
            for t in eval_args.utility_tasks:
                clean_utility_score[t] = utility_score[t]["acc,none"]
                print("Post utility score: ", clean_utility_score)
                wandb.log({"Post utility score": clean_utility_score})
        except Exception as e:
            print("Post utility score: ", utility_score)
            wandb.log({"Post utility score": utility_score})

    if eval_args.results_path:
        path = Path(eval_args.results_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2) + "\n")

    wandb.finish()

if __name__ == "__main__":
    main()
