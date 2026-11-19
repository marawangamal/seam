import itertools
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class DefenceSweepTest(unittest.TestCase):
    def test_axolotl_attack_sweep_resumes_and_preserves_subset_results(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts/tamia/run_defence_attack_eval.sbatch"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".venv/bin").mkdir(parents=True)
            fake = root / "fake.py"
            fake.write_text(f"#!{sys.executable}\n" """import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if Path(sys.argv[0]).name == 'uv':
    assert args == ['sync', '--frozen', '--only-group', 'attack']
    path = Path(os.environ['UV_PROJECT_ENVIRONMENT']) / 'bin'
    path.mkdir(parents=True, exist_ok=True)
    for name in ['python', 'axolotl']:
        if not (path / name).exists():
            (path / name).symlink_to(Path(__file__).resolve())
    raise SystemExit
if args[0] == '-c':
    print('sp')
    raise SystemExit
def arg(key): return args[args.index(key) + 1]
with open('calls.jsonl', 'a') as handle:
    handle.write(json.dumps(args) + '\\n')
if args[:2] == ['-m', 'src.train']:
    assert Path(sys.argv[0]).parent.parent.name == '.venv'
    assert arg('--alpha') == '5.0' and arg('--beta') == '1.0'
    assert arg('--harmful_response_separator') == ' '
    (Path(arg('--output_dir')) / 'checkpoint-500').mkdir(parents=True)
elif args[0] == 'train':
    assert Path(sys.argv[0]).parent.parent.name == '.venv-attack-123'
    assert args[1:4] == ['attack.yaml', '--launcher', 'python']
    assert Path(arg('--base-model')).is_dir()
    assert arg('--learning-rate') in ['5e-5', '8e-5', '1e-4', '2e-4']
    assert arg('--max-steps') in ['5', '10', '50', '100']
    Path(arg('--output-dir')).mkdir(parents=True)
else:
    assert args[:2] == ['-m', 'src.evaluation']
    assert Path(sys.argv[0]).parent.parent.name == '.venv-attack-123'
    assert '--utility-samples' not in args
    assert Path(arg('--model')).is_dir()
    path = Path(arg('--results-path'))
    if 'lr1e-4-ms10' in str(path) and not Path('failed-once').exists():
        Path('failed-once').touch()
        raise SystemExit(1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{}')
""")
            fake.chmod(0o755)
            (root / ".venv/bin/python").symlink_to(fake)
            uv = root / "uv"
            uv.symlink_to(fake)
            setup = root / "shell_setup.sh"
            setup.write_text(f'uv() {{ "{uv}" "$@"; }}\n')
            env = os.environ | {
                "BASH_ENV": str(setup),
                "SLURM_TMPDIR": directory,
                "SCRATCH": directory,
                "SLURM_JOB_ID": "123",
            }
            model = (
                "llama2-7b-beavertail--align-fft-seam--def-sp--coeff-sd1.0--coeff-up5.0"
            )
            names = [model] + [
                f"{model}--relearn-fft-att-sp-lr{lr}-ms5"
                for lr in ["5e-5", "8e-5", "1e-4", "2e-4"]
            ]
            originals = []
            for name in names:
                path = root / "artifacts/tamia/evals" / name / "results-1000.json"
                path.parent.mkdir(parents=True)
                path.write_text('{"preserved": true}')
                originals.append(path)
            old_logs = []
            for name in names[1:]:
                path = root / "artifacts/tamia/logs" / f"{name}.log"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("original subset sweep log")
                old_logs.append(path)
            command = [
                "bash",
                str(script),
                "defence.yaml",
                "attack.yaml",
                " ",
                "5.0",
                "1.0",
            ]
            first = subprocess.run(command, cwd=root, env=env, capture_output=True)
            self.assertNotEqual(first.returncode, 0)
            checkpoint = root / "artifacts/tamia/models" / model / "checkpoint-500"
            self.assertTrue(checkpoint.exists())
            subprocess.run(command, cwd=root, env=env, check=True, capture_output=True)
            calls = [
                json.loads(line)
                for line in (root / "calls.jsonl").read_text().splitlines()
            ]
            self.assertEqual(sum(call[:2] == ["-m", "src.train"] for call in calls), 1)
            attacks = [call for call in calls if call[0] == "train"]
            self.assertEqual(len(attacks), 16)
            self.assertEqual(
                {call[call.index("--max-steps") + 1] for call in attacks},
                {"5", "10", "50", "100"},
            )
            self.assertEqual(
                len(list(root.glob("artifacts/tamia/evals/*/results.json"))), 17
            )
            self.assertEqual(list((root / "artifacts/tamia/models").iterdir()), [])
            for path in originals:
                self.assertEqual(path.read_text(), '{"preserved": true}')
            for path in old_logs:
                self.assertEqual(path.read_text(), "original subset sweep log")
            self.assertEqual(
                len(list(root.glob("artifacts/tamia/logs/*--full-eval.log"))), 16
            )
            subprocess.run(command, cwd=root, env=env, check=True, capture_output=True)
            self.assertEqual(
                len((root / "calls.jsonl").read_text().splitlines()), len(calls)
            )

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
if Path(sys.argv[0]).name == 'uv':
    environment = Path(os.environ['UV_PROJECT_ENVIRONMENT'])
    group = args[-1]
    assert group == 'attack'
    assert args == ['sync', '--frozen', '--only-group', group]
    assert environment == Path(os.environ['SLURM_TMPDIR']) / ('.venv-' + group + '-' + os.environ['SLURM_JOB_ID'])
    assert os.environ['UV_OFFLINE'] == '1'
    (environment / 'bin').mkdir(parents=True, exist_ok=True)
    for name in ['python', 'axolotl']:
        executable = environment / 'bin' / name
        if not executable.exists():
            executable.symlink_to(Path(__file__).resolve())
    raise SystemExit
environment_name = Path(sys.argv[0]).parent.parent.name
if args[0] == '-c' or args[:2] == ['-m', 'src.train']:
    assert environment_name == '.venv'
else:
    assert environment_name == '.venv-attack-' + os.environ['SLURM_JOB_ID']
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
    assert str(Path.cwd()) in os.environ['PYTHONPATH'].split(os.pathsep)
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
            for name in ["python", "axolotl", "uv"]:
                (root / ".venv/bin" / name).symlink_to(fake)
            shell_setup = root / "shell_setup.sh"
            shell_setup.write_text(f'uv() {{ "{root / ".venv/bin/uv"}" "$@"; }}\n')
            env = os.environ | {
                "PATH": str(root / ".venv/bin") + os.pathsep + os.environ["PATH"],
                "SCRATCH": directory,
                "SLURM_TMPDIR": directory,
                "SLURM_JOB_ID": "123",
                "BASH_ENV": str(shell_setup),
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
