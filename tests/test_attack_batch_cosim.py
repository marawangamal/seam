"""Check attacks reuse retained defenses and clean up only evaluated attacks."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class AttackBatchCosimTests(unittest.TestCase):
    def test_paired_defenses_four_parallel_lrs_and_retention(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts/misc/run_attack_batch_cosim.sbatch"
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
    (path / 'python').symlink_to(Path(__file__).resolve())
    raise SystemExit
def arg(key): return args[args.index(key) + 1]
with open('calls.jsonl', 'a') as handle:
    handle.write(json.dumps({'args': args, 'gpu': os.environ['CUDA_VISIBLE_DEVICES']}) + '\\n')
if args[:2] == ['-m', 'src.eval']:
    assert Path(sys.argv[0]).parent.parent.name == '.venv'
    assert arg('--max_steps') == '250'
    assert arg('--attack_size') == '1000'
    assert arg('--gradient_cosim_steps') == '1'
    assert arg('--pre_utility') == arg('--post_utility') == 'False'
    assert Path(arg('--model_name'), 'config.json').exists()
    checkpoint = Path(arg('--output_dir')) / 'checkpoint-250'
    checkpoint.mkdir(parents=True)
    result = Path(arg('--results_path'))
    result.parent.mkdir(parents=True)
    result.write_text('{}')
    result.with_name('gradient_cosim.jsonl').write_text('{"step": 249}\\n')
elif args[:2] == ['-m', 'src.evaluation']:
    assert Path(sys.argv[0]).parent.parent.name.startswith('.venv-eval-')
    assert Path(arg('--model')).is_dir()
    Path(arg('--results-path')).write_text('{}')
else:
    raise AssertionError(args)
""")
            fake.chmod(0o755)
            (root / ".venv/bin/python").symlink_to(fake)
            shell_setup = root / "shell_setup.sh"
            uv = root / "uv"
            uv.symlink_to(fake)
            shell_setup.write_text(f'uv() {{ "{uv}" "$@"; }}\n')
            env = os.environ | {
                "BASH_ENV": str(shell_setup),
                "SCRATCH": directory,
                "SLURM_TMPDIR": directory,
            }
            for index, (separator, coefficient) in enumerate(
                [("nl", "0.01"), ("sp", "1.0")]
            ):
                model = (
                    "llama2-7b-beavertail--align-fft-seam--upstream-paper-config"
                    f"--def-{separator}--coeff-sd{coefficient}-rerun"
                )
                checkpoint = root / "artifacts/tamia/models" / model / "checkpoint-500"
                checkpoint.mkdir(parents=True)
                (checkpoint / "config.json").write_text("{}")
                result = subprocess.run(
                    ["bash", str(script)],
                    cwd=root,
                    env=env
                    | {
                        "SLURM_ARRAY_TASK_ID": str(index),
                        "SLURM_JOB_ID": str(index),
                    },
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue((checkpoint / "config.json").exists())
            calls = [
                json.loads(line)
                for line in (root / "calls.jsonl").read_text().splitlines()
            ]
            attacks = [call for call in calls if call["args"][1] == "src.eval"]
            evaluations = [
                call for call in calls if call["args"][1] == "src.evaluation"
            ]
            self.assertEqual(len(attacks), 8)
            self.assertEqual(len(evaluations), 8)
            self.assertEqual(sorted(call["gpu"] for call in attacks), list("00112233"))
            for call in attacks:
                args = call["args"]
                source = args[args.index("--model_name") + 1]
                separator = args[args.index("--harmful_response_separator") + 1]
                self.assertEqual(separator, "\n" if "--def-nl--" in source else " ")
                output = args[args.index("--output_dir") + 1]
                self.assertFalse((root / output).exists())
            self.assertEqual(
                len(list(root.glob("artifacts/tamia/evals/*/results.json"))), 8
            )


if __name__ == "__main__":
    unittest.main()
