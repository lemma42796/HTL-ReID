import contextlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SeedRelayTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module(
            "seed_relay_tested",
            REPO_ROOT / "tools" / "run_e071_e072_seed_relay.py",
        )

    def run_relay(self, results):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        output_root = Path(temporary.name) / "outputs"
        repo_root = Path(temporary.name) / "repo"
        output_root.mkdir()
        repo_root.mkdir()
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                self.module.subprocess, "check_output",
                return_value="test-commit\n"))
            run_one = stack.enter_context(mock.patch.object(
                self.module, "run_one", side_effect=results))
            stack.enter_context(mock.patch.object(sys, "argv", [
                "run_e071_e072_seed_relay.py",
                "--repo-root", str(repo_root),
                "--output-root", str(output_root),
            ]))
            returncode = self.module.main()
        chain_dir = output_root / self.module.CHAIN_NAME
        decision = json.loads(
            (chain_dir / "decision.json").read_text(encoding="utf-8"))
        return returncode, run_one, chain_dir, decision

    def test_seed3333_is_skipped_when_seed2222_exceeds_target(self):
        first = {
            "experiment": "E071",
            "status": "completed",
            "mAP": 70.01,
            "Rank1": 73.0,
        }
        returncode, run_one, chain_dir, decision = self.run_relay([first])

        self.assertEqual(returncode, 0)
        self.assertEqual(run_one.call_count, 1)
        self.assertTrue(decision["target_achieved"])
        self.assertTrue(decision["e072_skipped"])
        self.assertEqual(decision["selected_experiment"], "E071")
        self.assertTrue((chain_dir / "DONE").is_file())
        self.assertFalse((chain_dir / "FAILED").exists())

    def test_seed3333_runs_when_seed2222_misses_target(self):
        first = {
            "experiment": "E071",
            "status": "completed",
            "mAP": 69.99,
            "Rank1": 72.0,
        }
        second = {
            "experiment": "E072",
            "status": "completed",
            "mAP": 70.25,
            "Rank1": 73.5,
        }
        returncode, run_one, chain_dir, decision = self.run_relay(
            [first, second])

        self.assertEqual(returncode, 0)
        self.assertEqual(run_one.call_count, 2)
        self.assertTrue(decision["target_achieved"])
        self.assertFalse(decision["e072_skipped"])
        self.assertEqual(decision["selected_experiment"], "E072")
        self.assertEqual(decision["selected_mAP"], 70.25)
        self.assertTrue((chain_dir / "DONE").is_file())


if __name__ == "__main__":
    unittest.main()
