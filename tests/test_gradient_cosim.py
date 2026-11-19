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

from src.core.attack_trainer import MaskedAttackTrainer
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

    def test_attack_loss_and_gradients_match_defense_masked_ce(self):
        model = TinyLM().eval()
        trainer = MaskedAttackTrainer(
            model=model,
            train_dataset=self.harmful,
            align_dataset=self.refusal,
            args=TrainingArguments(
                output_dir=self.tmp.name, use_cpu=True, report_to=[]
            ),
            processing_class=SimpleNamespace(pad_token_id=0),
        )
        inputs = {key: value[:2].clone() for key, value in self.harmful[:].items()}
        inputs["labels"] = inputs["input_ids"].clone()
        inputs["refusal_input_ids"] = self.refusal[:2]["input_ids"]
        loss, outputs = trainer.compute_loss(model, inputs, return_outputs=True)
        actual_grad = torch.autograd.grad(loss, tuple(model.parameters()))

        labels = inputs["input_ids"].masked_fill(inputs["input_ids"] == 0, -100)
        mask = inputs["input_ids"] != inputs["refusal_input_ids"]
        author = object.__new__(SeamTrainer)
        expected_loss = author.masked_token_ce_loss(
            model(input_ids=inputs["input_ids"]).logits, labels, mask.float()
        )
        expected_grad = torch.autograd.grad(expected_loss, tuple(model.parameters()))
        torch.testing.assert_close(loss, expected_loss)
        for actual, expected in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual, expected)
        full_loss = author.ce_loss(outputs.logits, labels)
        self.assertFalse(torch.isclose(loss, full_loss).item())
        self.assertTrue(torch.equal(inputs["labels"], inputs["input_ids"]))

    def test_masked_attack_updates_match_harmful_only_sgd(self):
        torch.manual_seed(42)
        model = TinyLM()
        model.dropout.p = 0
        expected = copy.deepcopy(model)
        trainer = MaskedAttackTrainer(
            model=model,
            args=TrainingArguments(
                output_dir=self.tmp.name,
                max_steps=1,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=2,
                max_grad_norm=0,
                save_strategy="no",
                report_to=[],
                use_cpu=True,
                disable_tqdm=True,
                dataloader_pin_memory=False,
            ),
            processing_class=SimpleNamespace(pad_token_id=0),
            train_dataset=self.harmful,
            align_dataset=self.refusal,
            optimizers=(torch.optim.SGD(model.parameters(), lr=0.1), None),
        )
        optimizer = torch.optim.SGD(expected.parameters(), lr=0.1)
        batch = self.harmful[:]
        mask = batch["input_ids"] != self.refusal[:]["input_ids"]
        loss = SeamTrainer.masked_token_ce_loss(
            None, expected(**batch).logits, batch["labels"], mask.float()
        )
        loss.backward()
        optimizer.step()
        trainer.train()
        for actual, target in zip(model.parameters(), expected.parameters()):
            torch.testing.assert_close(actual, target)

    def test_batch_cosines_use_current_samples_and_unscaled_microbatch_gradients(self):
        model = TinyLM()
        model.dropout.p = 0
        benign_rows = self.benign.to_dict()
        for index in range(4):
            benign_rows["input_ids"][index][2] = 8 + index
            benign_rows["labels"][index][2] = 8 + index
        benign = Dataset.from_dict(benign_rows).with_format("torch")
        trainer = MaskedAttackTrainer(
            model=model,
            args=TrainingArguments(
                output_dir=self.tmp.name,
                max_steps=2,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=2,
                max_grad_norm=0,
                save_strategy="no",
                report_to=[],
                use_cpu=True,
                disable_tqdm=True,
                dataloader_pin_memory=False,
            ),
            processing_class=SimpleNamespace(pad_token_id=0),
            train_dataset=self.harmful,
            align_dataset=self.refusal,
            benign_dataset=benign,
            gradient_cosim_steps=1,
            gradient_cosim_path=self.path,
            optimizers=(torch.optim.SGD(model.parameters(), lr=0), None),
        )
        trainer.train()
        rows = [json.loads(line) for line in self.path.read_text().splitlines()]
        self.assertEqual([row["step"] for row in rows], [0, 0, 1, 1])
        self.assertEqual([row["microbatch"] for row in rows], [1, 2, 3, 4])
        self.assertEqual(
            sorted(row["sample_indices"][0] for row in rows), list(range(4))
        )
        for row in rows:
            index = row["sample_indices"][0]
            harmful_batch = self.harmful[index : index + 1]
            benign_batch = benign[index : index + 1]
            mask = (
                harmful_batch["input_ids"]
                != self.refusal[index : index + 1]["input_ids"]
            )
            harmful_loss = SeamTrainer.masked_token_ce_loss(
                None,
                model(**harmful_batch).logits,
                harmful_batch["labels"],
                mask.float(),
            )
            benign_loss = SeamTrainer.ce_loss(
                None, model(**benign_batch).logits, benign_batch["labels"]
            )
            harmful_grad = torch.autograd.grad(harmful_loss, tuple(model.parameters()))
            benign_grad = torch.autograd.grad(benign_loss, tuple(model.parameters()))
            h = torch.cat([g.flatten().double() for g in harmful_grad])
            b = torch.cat([g.flatten().double() for g in benign_grad])
            expected = (h @ b / (h.norm() * b.norm() + 1e-8)).item()
            self.assertAlmostEqual(row["cosine_similarity"], expected, places=6)
            self.assertAlmostEqual(
                row["harmful_gradient_norm"], h.norm().item(), places=5
            )
            self.assertAlmostEqual(
                row["benign_gradient_norm"], b.norm().item(), places=5
            )
            self.assertAlmostEqual(row["harmful_loss"], harmful_loss.item(), places=6)
            self.assertAlmostEqual(row["benign_loss"], benign_loss.item(), places=6)
        self.assertIn("attack/cosine_similarity", self.log.call_args.args[0])

    def test_batch_cosine_logging_preserves_harmful_updates_optimizer_and_rng(self):
        torch.manual_seed(42)
        initial = TinyLM()
        runs = []
        for enabled in [False, True]:
            model = copy.deepcopy(initial)
            trainer = MaskedAttackTrainer(
                model=model,
                args=TrainingArguments(
                    output_dir=self.tmp.name,
                    max_steps=3,
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=2,
                    save_strategy="no",
                    report_to=[],
                    use_cpu=True,
                    disable_tqdm=True,
                    dataloader_pin_memory=False,
                    seed=19,
                ),
                processing_class=SimpleNamespace(pad_token_id=0),
                train_dataset=self.harmful,
                align_dataset=self.refusal,
                benign_dataset=self.benign if enabled else None,
                gradient_cosim_steps=int(enabled),
                gradient_cosim_path=self.path if enabled else None,
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
            torch.testing.assert_close(value, runs[1][0][name], rtol=0, atol=0)
        for index, state in runs[0][1]["state"].items():
            for key, value in state.items():
                other = runs[1][1]["state"][index][key]
                if torch.is_tensor(value):
                    torch.testing.assert_close(value, other, rtol=0, atol=0)
                else:
                    self.assertEqual(value, other)
        self.assertTrue(torch.equal(runs[0][2], runs[1][2]))

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
