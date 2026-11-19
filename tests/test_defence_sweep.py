import itertools
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class DefenceSweepTest(unittest.TestCase):
    def test_grid_commands_result_retention_and_skipping(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts/tamia/run_defence_sweep.sbatch"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".venv/bin").mkdir(parents=True)
            (root / ".venv/bin/activate").touch()
            fake = root / "fake.py"
            fake.write_text(f"#!{sys.executable}\n" """import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if args[0] == '-c':
    print('sp')
    raise SystemExit
def arg(key): return args[args.index(key) + 1]
with open('calls.jsonl', 'a') as handle:
    handle.write(json.dumps(args) + '\\n')
if args[:2] == ['-m', 'src.train']:
    assert arg('--config') == 'defence.yaml'
    path = Path(arg('--output_dir')) / 'checkpoint-500'
    path.mkdir(parents=True)
elif args[0] == 'train':
    assert args[1] == 'attack.yaml'
    assert Path(arg('--base-model')).is_dir()
    assert arg('--max-steps') == '5'
    Path(arg('--output-dir')).mkdir(parents=True)
else:
    assert args[:2] == ['-m', 'src.evaluation']
    assert Path(arg('--model')).is_dir()
    assert arg('--utility-samples') == '1000'
    path = Path(arg('--results-path'))
    assert path.parts[:3] == ('artifacts', 'tamia', 'evals')
    path.parent.mkdir(parents=True)
    path.write_text('{}')
""")
            fake.chmod(0o755)
            for name in ["python", "axolotl"]:
                (root / ".venv/bin" / name).symlink_to(fake)
            env = os.environ | {
                "PATH": str(root / ".venv/bin") + os.pathsep + os.environ["PATH"],
                "SCRATCH": directory,
            }
            pairs = itertools.product(
                ["0.1", "1.0", "5.0"], ["0.01", "0.1", "1.0", "5.0"]
            )
            for index, (alpha, beta) in enumerate(pairs):
                with self.subTest(index=index):
                    command = ["bash", str(script), "defence.yaml", "attack.yaml", " "]
                    task_env = env | {"SLURM_ARRAY_TASK_ID": str(index)}
                    subprocess.run(
                        command, cwd=root, env=task_env, check=True, capture_output=True
                    )
                    calls = [
                        json.loads(line)
                        for line in (root / "calls.jsonl").read_text().splitlines()
                    ]
                    train = calls[-10]
                    self.assertEqual(train[:2], ["-m", "src.train"])
                    self.assertEqual(train[train.index("--alpha") + 1], alpha)
                    self.assertEqual(train[train.index("--beta") + 1], beta)
                    self.assertEqual(
                        train[train.index("--harmful_response_separator") + 1], " "
                    )
                    count = len(calls)
                    subprocess.run(
                        command, cwd=root, env=task_env, check=True, capture_output=True
                    )
                    self.assertEqual(
                        len((root / "calls.jsonl").read_text().splitlines()), count
                    )
            self.assertEqual(
                len(list((root / "artifacts/tamia/evals").glob("*/results-1000.json"))),
                60,
            )
            self.assertEqual(list((root / "artifacts/tamia/models").iterdir()), [])
