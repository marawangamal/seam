"""Exercise the evaluation entry point without loading models or calling APIs."""

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.mark.parametrize("skip_gpt", [False, True])
@pytest.mark.parametrize("save_results", [False, True])
def test_eval_gpt_and_local_results(monkeypatch, tmp_path, skip_gpt, save_results):
    metrics = ModuleType("src.utils.metrics")
    metrics.eval_harmfulness = MagicMock(return_value=0.25)
    metrics.eval_harmfulness_gpt = MagicMock(return_value=0.5)
    scores = {"arc_easy": {"acc,none": 0.4, "acc_norm,none": 0.6}}
    metrics.eval_utility = MagicMock(return_value=scores)
    datasets = ModuleType("src.data_processing._datasets")
    datasets.construct_beavertails_dataset = MagicMock(return_value=([], [], []))
    config = ModuleType("src.utils.config")
    config.load_yaml_config = lambda path: {}
    transformers = MagicMock()
    wandb = MagicMock()
    for name, module in {
        "torch": MagicMock(),
        "wandb": wandb,
        "peft": MagicMock(),
        "transformers": transformers,
        "src.data_processing._datasets": datasets,
        "src.utils.config": config,
        "src.utils.metrics": metrics,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    config.MyParser = MagicMock()
    path = Path(__file__).resolve().parents[1] / "src/eval.py"
    spec = importlib.util.spec_from_file_location("seam_eval_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    assert module.EvalArguments().skip_gpt is False
    assert module.EvalArguments().results_path is None

    @dataclass
    class TrainingArguments:
        per_device_train_batch_size: int = 4

    output = tmp_path / "nested/results.json"
    args = module.EvalArguments(
        model_name="model",
        utility_tasks=["arc_easy"],
        skip_gpt=skip_gpt,
        results_path=str(output) if save_results else None,
    )
    config.MyParser.return_value.parse_args_into_dataclasses.return_value = (
        TrainingArguments(),
        args,
    )
    monkeypatch.setattr(sys, "argv", ["eval"])
    module.main()
    transformers.Trainer.return_value.train.assert_called_once_with()
    assert metrics.eval_harmfulness_gpt.call_count == (0 if skip_gpt else 1)
    assert metrics.eval_utility.call_count == 2
    if save_results:
        results = json.loads(output.read_text())
        assert results["model_name"] == "model"
        assert results["harmfulness"] == 0.25
        assert results["pre_utility"] == scores
        assert results["post_utility"] == scores
        if skip_gpt:
            assert "harmfulness_gpt" not in results
        else:
            assert results["harmfulness_gpt"] == 0.5
    else:
        assert not output.exists()
    wandb.log.assert_any_call({"Harmful score": 0.25})
    wandb.finish.assert_called_once_with()
