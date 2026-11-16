"""Check gradient probes without downloading models or using GPUs."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from datasets import Dataset
from torch import nn
from transformers import PretrainedConfig, Trainer, TrainingArguments
from transformers.modeling_outputs import CausalLMOutput

from src.core.gradient_cosim import GradientCosimCallback
from src.core.trainer import SeamTrainer


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = PretrainedConfig()
        self.embedding = nn.Embedding(13, 8)
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(8, 13, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        logits = self.head(self.dropout(self.embedding(input_ids)))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, 13), labels[:, 1:].reshape(-1)
            )
        return CausalLMOutput(loss=loss, logits=logits)


def dataset(tokens):
    return Dataset.from_dict(
        {
            "input_ids": [tokens] * 4,
            "attention_mask": [[int(t != 0) for t in tokens]] * 4,
            "labels": [[t if t else -100 for t in tokens]] * 4,
        }
    ).with_format("torch")


class GradientCosimTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "gradient_cosim.jsonl"
        self.harmful = dataset([1, 2, 3, 4, 0, 0])
        self.refusal = dataset([1, 2, 5, 6, 0, 0])
        self.benign = dataset([1, 7, 8, 9, 0, 0])
        self.log = patch("src.core.gradient_cosim.wandb.log").start()
        self.define = patch("src.core.gradient_cosim.wandb.define_metric").start()

    def tearDown(self):
        patch.stopall()
        self.tmp.cleanup()
        torch.set_num_threads(self.threads)

    def callback(self, interval=1):
        return GradientCosimCallback(
            self.harmful,
            self.refusal,
            self.benign,
            0,
            self.path,
            interval=interval,
            samples=2,
        )

    def test_matches_defense_losses_and_preserves_gradients_rng_and_mode(self):
        torch.manual_seed(7)
        model = TinyLM().train()
        callback = self.callback()
        for param in model.parameters():
            param.grad = torch.randn_like(param)
        before = [p.grad.clone() for p in model.parameters()]
        rng = torch.get_rng_state().clone()
        args = SimpleNamespace(bf16=False, fp16=False)
        callback.on_train_begin(args, SimpleNamespace(global_step=0), None, model=model)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for param, grad in zip(model.parameters(), before):
            self.assertTrue(torch.equal(param.grad, grad))

        model.eval()
        author = object.__new__(SeamTrainer)
        harmful_loss = author.masked_token_ce_loss(
            model(**callback.harmful).logits,
            callback.harmful["labels"],
            callback.mask.float(),
        )
        harmful_grad = torch.autograd.grad(harmful_loss, tuple(model.parameters()))
        benign_loss = author.ce_loss(
            model(**callback.benign).logits, callback.benign["labels"]
        )
        benign_grad = torch.autograd.grad(benign_loss, tuple(model.parameters()))
        h = torch.cat([g.flatten().double() for g in harmful_grad])
        b = torch.cat([g.flatten().double() for g in benign_grad])
        expected = (h @ b / (h.norm() * b.norm() + 1e-8)).item()
        row = json.loads(self.path.read_text())
        self.assertAlmostEqual(row["cosine_similarity"], expected, places=6)
        self.assertAlmostEqual(row["harmful_probe_loss"], harmful_loss.item(), places=6)
        self.assertAlmostEqual(row["benign_probe_loss"], benign_loss.item(), places=6)
        self.assertEqual(row["probe"]["indices"], [0, 1])
        self.assertIn("attack/cosine_similarity", self.log.call_args.args[0])

    def test_interval_and_final_step_are_saved_once(self):
        callback = self.callback(interval=2)
        model = TinyLM()
        args = SimpleNamespace(bf16=False, fp16=False)
        state = SimpleNamespace(global_step=0)
        callback.on_train_begin(args, state, None, model=model)
        for step in [1, 2, 3]:
            state.global_step = step
            callback.on_step_end(args, state, None, model=model)
        callback.on_train_end(args, state, None, model=model)
        callback.on_train_end(args, state, None, model=model)
        rows = [json.loads(line) for line in self.path.read_text().splitlines()]
        self.assertEqual([r["step"] for r in rows], [0, 2, 3])

    def test_zero_gradients_are_finite(self):
        model = TinyLM()
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()
        callback = self.callback()
        callback.on_train_begin(
            SimpleNamespace(bf16=False, fp16=False),
            SimpleNamespace(global_step=0),
            None,
            model=model,
        )
        row = json.loads(self.path.read_text())
        self.assertEqual(row["cosine_similarity"], 0)
        self.assertEqual(row["harmful_gradient_norm"], 0)
        self.assertEqual(row["benign_gradient_norm"], 0)

    def test_training_updates_optimizer_and_rng_are_unchanged(self):
        torch.manual_seed(42)
        initial = TinyLM()
        runs = []
        for enabled in [False, True]:
            model = copy.deepcopy(initial)
            args = TrainingArguments(
                output_dir=str(Path(self.tmp.name) / str(enabled)),
                max_steps=3,
                per_device_train_batch_size=2,
                save_strategy="no",
                report_to=[],
                use_cpu=True,
                disable_tqdm=True,
                dataloader_pin_memory=False,
                seed=19,
            )
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=self.harmful,
                callbacks=[self.callback()] if enabled else [],
            )
            trainer.train()
            runs.append(
                (
                    copy.deepcopy(model.state_dict()),
                    copy.deepcopy(trainer.optimizer.state_dict()),
                    torch.get_rng_state().clone(),
                )
            )
        for name, value in runs[0][0].items():
            self.assertTrue(torch.equal(value, runs[1][0][name]), name)
        for index, state in runs[0][1]["state"].items():
            for key, value in state.items():
                other = runs[1][1]["state"][index][key]
                if torch.is_tensor(value):
                    self.assertTrue(torch.equal(value, other), key)
                else:
                    self.assertEqual(value, other)
        self.assertTrue(torch.equal(runs[0][2], runs[1][2]))
        rows = [json.loads(line) for line in self.path.read_text().splitlines()]
        self.assertEqual([r["step"] for r in rows], [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
