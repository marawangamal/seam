"""Check attack sample selection through the evaluation entry point."""

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, mock_open, patch

from datasets import Dataset

from src import eval as evaluation


class DatasetReady(Exception):
    """Stop before training or evaluation starts."""


class EvalDatasetTests(unittest.TestCase):
    def test_requested_attack_size_reaches_trainer(self):
        with tempfile.TemporaryDirectory() as output:
            for attack_mode, size, expected in [
                (True, 10, 10),
                (True, 100, 100),
                (True, 200, 200),
                (True, 500, 500),
                (True, 1000, 1000),
                (False, 10, 1000),
            ]:
                with self.subTest(attack_mode=attack_mode, size=size):
                    harmful = Dataset.from_dict(
                        {"input_ids": [[i] for i in range(1100)]}
                    )
                    refusal = Dataset.from_dict(
                        {"input_ids": [[i] for i in range(1100)]}
                    )
                    test = Dataset.from_dict({"input_ids": [[i] for i in range(120)]})
                    train_source = SimpleNamespace(
                        column_names=[], map=Mock(side_effect=[harmful, refusal])
                    )
                    test_source = SimpleNamespace(
                        column_names=[], map=Mock(return_value=test)
                    )
                    config = {
                        "TrainingArguments": {
                            "output_dir": output,
                            "use_cpu": True,
                            "report_to": [],
                        },
                        "EvalArguments": {
                            "model_name": "unused",
                            "pre_utility": False,
                            "attack_dataset": attack_mode,
                        },
                    }
                    tokenizer = SimpleNamespace(eos_token="eos", padding_side="right")
                    with (
                        patch("sys.argv", ["eval.py", "--attack_size", str(size)]),
                        patch.object(
                            evaluation, "load_yaml_config", return_value=config
                        ),
                        patch.object(evaluation, "wandb"),
                        patch.object(
                            evaluation.AutoTokenizer,
                            "from_pretrained",
                            return_value=tokenizer,
                        ),
                        patch.object(
                            evaluation.AutoModelForCausalLM, "from_pretrained"
                        ),
                        patch(
                            "src.data_processing._datasets.open",
                            mock_open(read_data="[]"),
                        ),
                        patch(
                            "src.data_processing._datasets.Dataset.from_list",
                            return_value=train_source,
                        ),
                        patch(
                            "src.data_processing._datasets.load_dataset",
                            return_value=test_source,
                        ),
                        patch.object(
                            evaluation, "Trainer", side_effect=DatasetReady
                        ) as trainer,
                        self.assertRaises(DatasetReady),
                    ):
                        evaluation.main()
                    selected = trainer.call_args.kwargs["train_dataset"]
                    self.assertEqual(len(selected), expected)
                    self.assertEqual(selected[0]["input_ids"].item(), 0)
                    self.assertEqual(selected[-1]["input_ids"].item(), expected - 1)


if __name__ == "__main__":
    unittest.main()
