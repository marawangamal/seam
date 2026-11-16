"""Observe defense-style gradient alignment during ordinary attack training."""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from transformers import TrainerCallback


class GradientCosimCallback(TrainerCallback):
    def __init__(
        self, harmful, refusal, benign, pad_token_id, path, interval=1, samples=4
    ):
        self.path = Path(path)
        self.interval = interval
        self.last_step = None
        if interval < 1 or samples < 1:
            raise ValueError("Cosine interval and probe size must be positive")
        if min(len(harmful), len(refusal), len(benign)) < samples:
            raise ValueError("Probe size exceeds the available examples")
        self.probe = {
            "indices": list(range(samples)),
            "harmful_fingerprint": harmful._fingerprint,
            "benign_fingerprint": benign._fingerprint,
            "harmful_loss": "SEAM attack CE masked by harmful/refusal token differences",
            "benign_loss": "Alpaca full-sequence CE",
            "model_mode": "eval",
        }
        self.harmful, self.refusal, self.benign = [
            {
                key: torch.stack([dataset[i][key].clone() for i in range(samples)])
                for key in ["input_ids", "attention_mask"]
            }
            for dataset in [harmful, refusal, benign]
        ]
        self.mask = self.harmful["input_ids"] != self.refusal["input_ids"]
        for batch in [self.harmful, self.benign]:
            batch["labels"] = batch["input_ids"].clone()
            batch["labels"][batch["labels"] == pad_token_id] = -100

    def _gradients(self, model, parameters, batch, device, mask=None):
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        ).logits
        logits = logits[..., :-1, :].contiguous()
        labels = batch["labels"][..., 1:].contiguous()
        if mask is not None:
            # Match SeamTrainer.masked_token_ce_loss, including both shifted masks.
            mask = mask.to(device)
            logits = logits * mask[..., :-1].unsqueeze(-1)
            labels = labels * mask[..., 1:]
            labels = labels.masked_fill(labels == 0, -100)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        return loss.item(), [
            grad.detach() if grad is not None else None for grad in gradients
        ]

    def _record(self, args, state, model):
        if self.last_step == state.global_step:
            return
        parameters = [param for param in model.parameters() if param.requires_grad]
        device = parameters[0].device
        devices = [device.index] if device.type == "cuda" else []
        training = model.training
        dtype = torch.bfloat16 if args.bf16 else torch.float16
        try:
            model.eval()
            with torch.random.fork_rng(devices=devices), torch.enable_grad():
                with torch.autocast(
                    device.type, dtype=dtype, enabled=args.bf16 or args.fp16
                ):
                    harmful_loss, harmful_grad = self._gradients(
                        model, parameters, self.harmful, device, self.mask
                    )
                    harmful_grad = [
                        grad.cpu() if grad is not None else None
                        for grad in harmful_grad
                    ]
                    benign_loss, benign_grad = self._gradients(
                        model, parameters, self.benign, device
                    )
        finally:
            model.train(training)

        # Reduce on the GPU, transferring the offloaded gradient one tensor at a time.
        # This avoids both CPU full-model reductions and two full GPU gradient copies.
        products = torch.zeros(3, device=device, dtype=torch.float32)
        for harmful, benign in zip(harmful_grad, benign_grad):
            if harmful is not None:
                harmful = harmful.to(device).float()
                products[1] += harmful.square().sum()
            if benign is not None:
                benign = benign.float()
                products[2] += benign.square().sum()
            if harmful is not None and benign is not None:
                products[0] += (harmful * benign).sum()
        dot, harmful_squared, benign_squared = products.cpu().tolist()
        harmful_norm = math.sqrt(harmful_squared)
        benign_norm = math.sqrt(benign_squared)
        cosine = dot / (harmful_norm * benign_norm + 1e-8)
        metrics = {
            "cosine_similarity": cosine,
            "harmful_gradient_norm": harmful_norm,
            "benign_gradient_norm": benign_norm,
            "harmful_probe_loss": harmful_loss,
            "benign_probe_loss": benign_loss,
        }
        metrics = {
            key: value if math.isfinite(value) else None
            for key, value in metrics.items()
        }
        row = {"step": state.global_step, **metrics}
        if self.last_step is None:
            row["probe"] = self.probe
        with self.path.open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        wandb.log(
            {"attack/step": state.global_step}
            | {f"attack/{key}": value for key, value in metrics.items()}
        )
        self.last_step = state.global_step

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")
        wandb.define_metric("attack/step")
        wandb.define_metric("attack/*", step_metric="attack/step")
        self._record(args, state, model)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.interval == 0:
            self._record(args, state, model)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        self._record(args, state, model)
