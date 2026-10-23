"""Check relocated launchers without loading checkpoints or starting GPU jobs."""

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class ServerConfigTest(unittest.TestCase):
    def test_default_config_points_to_repository(self):
        from app.recovery_server import parse_args

        with patch.object(sys, "argv", ["recovery_server", "--checkpoint", "unused.pt", "--bank-dir", "unused"]):
            config = parse_args().intent_config
        self.assertEqual(config, ROOT / "configs/libero.yaml")
        self.assertTrue(config.is_file())


class LauncherTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="resettle entrypoints ")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.foreign_cwd = self.directory / "outside-repository"
        self.foreign_cwd.mkdir()
        self.calls = self.directory / "calls"
        self.calls.mkdir()
        self.data = self.directory / "data"
        for suite in ("goal", "spatial", "object", "10"):
            (self.data / f"libero_{suite}_no_noops_1.0.0_lerobot").mkdir(parents=True)
        self.checkpoint = self.directory / "latest.pt"
        self.checkpoint.touch()
        shutil.copyfile(ROOT / "configs/libero.yaml", self.directory / "params-pretrain.yaml")
        self.bank = self.directory / "bank"
        self.output = self.directory / "training-output"

        # Run the launchers' real configuration code; intercept only GPU discovery
        # and module execution. Separate records avoid concurrent bank-worker writes.
        recorder = self.directory / "record_python.py"
        recorder.write_text(textwrap.dedent('''\
            import json
            import os
            import sys
            from pathlib import Path

            import yaml

            arguments = sys.argv[1:]
            if arguments[0] == "-":
                sys.argv = arguments
                exec(compile(sys.stdin.read(), "<launcher-config>", "exec"), {"__name__": "__main__"})
            elif arguments == ["-c", "import torch; print(torch.cuda.device_count())"]:
                print(4)
            elif "-m" in arguments:
                module = arguments[arguments.index("-m") + 1]
                if module not in {"app.main", "app.build_bank", "app.recovery_server"}:
                    raise SystemExit(f"Unexpected module: {module}")
                record = {
                    "cwd": os.getcwd(), "argv": arguments,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                }
                if module == "app.main":
                    config = Path(arguments[arguments.index("--fname") + 1])
                    record["config"] = yaml.safe_load(config.read_text())
                destination = Path(os.environ["RESETTLE_TEST_CALLS"]) / f"{os.getpid()}.json"
                destination.write_text(json.dumps(record))
            else:
                raise SystemExit(f"Unexpected Python invocation: {arguments}")
            '''))
        fake_python = self.directory / "fake-python"
        fake_python.write_text(
            "#!/usr/bin/env bash\n"
            f"exec {shlex.quote(sys.executable)} {shlex.quote(str(recorder))} \"$@\"\n"
        )
        fake_python.chmod(0o755)
        self.environment = {
            **os.environ,
            "PYTHON": str(fake_python),
            "RESETTLE_TEST_CALLS": str(self.calls),
            "LIBERO_DATA_ROOT": str(self.data),
            "VJEPA_PRETRAIN": str(self.checkpoint),
            "RESETTLE_CHECKPOINT": str(self.checkpoint),
            "RESETTLE_BANK": str(self.bank),
            "RESETTLE_OUTPUT": str(self.output),
            "NUM_GPUS": "2",
            "BANK_BATCH_SIZE": "16",
        }

    def run_launcher(self, name, *arguments):
        result = subprocess.run(
            ["bash", str(ROOT / "scripts" / name), *arguments],
            cwd=self.foreign_cwd, env=self.environment,
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = [json.loads(path.read_text()) for path in self.calls.glob("*.json")]
        self.assertTrue(calls, "Launcher never invoked its Python module")
        for call in calls:
            self.assertEqual(Path(call["cwd"]).resolve(), ROOT)
        return calls

    def test_training_launcher_from_another_directory(self):
        calls = self.run_launcher("train.sh")
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(call["argv"][:2], ["-m", "app.main"])
        self.assertEqual(call["argv"][call["argv"].index("--devices") + 1:], ["cuda:0", "cuda:1"])
        self.assertEqual(call["config"]["folder"], str(self.output))
        self.assertEqual(call["config"]["meta"]["pretrain_checkpoint"], str(self.checkpoint))
        self.assertEqual(call["config"]["data"]["datasets"], [
            str(self.data / f"libero_{suite}_no_noops_1.0.0_lerobot")
            for suite in ("object", "goal", "spatial", "10")
        ])

    def test_bank_launcher_from_another_directory(self):
        calls = self.run_launcher("build_bank.sh")
        workers = [call for call in calls if "--suite" in call["argv"]]
        mergers = [call for call in calls if "--merge" in call["argv"]]
        self.assertEqual(len(calls), 5)
        self.assertEqual(len(workers), 4)
        self.assertEqual(len(mergers), 1)
        self.assertEqual(mergers[0]["argv"][:2], ["-m", "app.build_bank"])
        expected_devices = {suite: str(rank) for rank, suite in enumerate(("goal", "spatial", "object", "10"))}
        observed_devices = {}
        for call in workers:
            arguments = call["argv"]
            self.assertEqual(arguments[:3], ["-u", "-m", "app.build_bank"])
            suite = arguments[arguments.index("--suite") + 1]
            observed_devices[suite] = call["cuda_visible_devices"]
            self.assertEqual(arguments[arguments.index("--data-root") + 1], str(self.data))
            self.assertEqual(arguments[arguments.index("--output-dir") + 1], str(self.bank / f"libero_{suite}"))
        self.assertEqual(observed_devices, expected_devices)

    def test_server_launcher_from_another_directory(self):
        calls = self.run_launcher("serve.sh", "--host", "0.0.0.0", "--port", "18083")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["argv"], [
            "-m", "app.recovery_server",
            "--checkpoint", str(self.checkpoint), "--bank-dir", str(self.bank),
            "--host", "0.0.0.0", "--port", "18083",
        ])


if __name__ == "__main__":
    unittest.main()
