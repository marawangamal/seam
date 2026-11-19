"""Train masked harmful CE and observe alignment with current Alpaca batches."""

import json
import math
from pathlib import Path

import torch
import wandb

from transformers import Trainer

from src.core.trainer import SeamTrainer


class MaskedAttackTrainer(Trainer):
    def __init__(
        self,
        *args,
        align_dataset,
        benign_dataset=None,
        gradient_cosim_steps=0,
        gradient_cosim_path=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False
        self.gradient_cosim_steps = gradient_cosim_steps
        self.gradient_cosim_path = (
            Path(gradient_cosim_path) if gradient_cosim_path is not None else None
        )
        self.last_cosim_step = None
        self.microbatch = 0
        if len(self.train_dataset) != len(align_dataset):
            raise ValueError("Harmful and refusal datasets must have equal lengths")
        self.train_dataset = self.train_dataset.add_column(
            "refusal_input_ids", align_dataset["input_ids"].tolist()
        )
        if gradient_cosim_steps:
            if benign_dataset is None or self.gradient_cosim_path is None:
                raise ValueError("Batch cosine logging requires Alpaca data and a path")
            if len(benign_dataset) != len(self.train_dataset):
                raise ValueError("Harmful and benign datasets must have equal lengths")
            for key in ["input_ids", "attention_mask", "labels"]:
                self.train_dataset = self.train_dataset.add_column(
                    f"benign_{key}", benign_dataset[key].tolist()
                )
            self.train_dataset = self.train_dataset.add_column(
                "sample_index", list(range(len(self.train_dataset)))
            )
            self.gradient_cosim_path.parent.mkdir(parents=True, exist_ok=True)
            self.gradient_cosim_path.write_text("")
            wandb.define_metric("attack/step")
            wandb.define_metric("attack/*", step_metric="attack/step")

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        for key in [
            "refusal_input_ids",
            "benign_input_ids",
            "benign_attention_mask",
            "benign_labels",
            "sample_index",
        ]:
            if key not in self._signature_columns:
                self._signature_columns.append(key)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        mask = inputs["input_ids"] != inputs["refusal_input_ids"]
        labels = inputs["labels"].masked_fill(
            inputs["input_ids"] == self.processing_class.pad_token_id, -100
        )
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        loss = SeamTrainer.masked_token_ce_loss(
            self, outputs.logits, labels, mask.float()
        )
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        record = (
            self.gradient_cosim_steps > 0
            and self.state.global_step % self.gradient_cosim_steps == 0
        )
        self.microbatch += 1
        if not record:
            return super().training_step(model, inputs, num_items_in_batch)

        model.train()
        inputs = self._prepare_inputs(inputs)
        parameters = [p for p in model.parameters() if p.requires_grad]
        device = parameters[0].device
        devices = [device.index] if device.type == "cuda" else []
        # Observe the paired benign batch without changing .grad or attack RNG.
        with (
            torch.random.fork_rng(devices=devices),
            self.compute_loss_context_manager(),
        ):
            outputs = model(
                input_ids=inputs["benign_input_ids"],
                attention_mask=inputs["benign_attention_mask"],
            )
            benign_loss = SeamTrainer.ce_loss(
                self, outputs.logits, inputs["benign_labels"]
            )
            benign_grad = torch.autograd.grad(
                benign_loss, parameters, allow_unused=True
            )
        benign_grad = [g.cpu() if g is not None else None for g in benign_grad]
        del outputs

        # Preserve earlier microbatch gradients when gradient accumulation is used.
        previous = [
            p.grad.detach().cpu().clone() if p.grad is not None else None
            for p in parameters
        ]
        loss = super().training_step(model, inputs, num_items_in_batch)
        scale = self.accelerator.scaler.get_scale() if self.accelerator.scaler else 1
        products = torch.zeros(3, device=device, dtype=torch.float32)
        for param, before, benign in zip(parameters, previous, benign_grad):
            harmful = None
            if param.grad is not None:
                harmful = param.grad.detach().float()
                if before is not None:
                    harmful = harmful - before.to(device).float()
                harmful = harmful * (self.args.gradient_accumulation_steps / scale)
                products[1] += harmful.square().sum()
            if benign is not None:
                benign = benign.to(device).float()
                products[2] += benign.square().sum()
            if harmful is not None and benign is not None:
                products[0] += (harmful * benign).sum()
        dot, harmful_squared, benign_squared = products.cpu().tolist()
        harmful_norm = math.sqrt(harmful_squared)
        benign_norm = math.sqrt(benign_squared)
        metrics = {
            "cosine_similarity": dot / (harmful_norm * benign_norm + 1e-8),
            "harmful_gradient_norm": harmful_norm,
            "benign_gradient_norm": benign_norm,
            "harmful_loss": loss.item() * self.args.gradient_accumulation_steps,
            "benign_loss": benign_loss.item(),
        }
        metrics = {
            key: value if math.isfinite(value) else None
            for key, value in metrics.items()
        }
        row = {
            "step": self.state.global_step,
            "microbatch": self.microbatch,
            "sample_indices": inputs["sample_index"].tolist(),
            **metrics,
        }
        with self.gradient_cosim_path.open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        wandb.log(
            {"attack/step": self.state.global_step}
            | {f"attack/{key}": value for key, value in metrics.items()}
        )
        self.last_cosim_step = self.state.global_step
        return loss
