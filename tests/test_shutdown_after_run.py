import contextlib
import importlib.util
import subprocess
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


class ShutdownAfterRunTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module(
            "shutdown_after_run_tested",
            REPO_ROOT / "tools" / "shutdown_after_run.py",
        )

    def test_poweroff_calls_shutdown_only_after_all_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "outputs"
            output_dir = root / "relay"
            output_dir.mkdir(parents=True)
            marker = output_dir / "DONE"
            marker.write_text("done\n", encoding="utf-8")
            fake_shutdown = Path(temporary) / "shutdown"
            fake_shutdown.write_text("#!/bin/sh\n", encoding="utf-8")
            fake_lock = mock.Mock()
            events = []

            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(
                    self.module, "OUTPUT_ROOT", root.resolve()))
                stack.enter_context(mock.patch.object(
                    self.module, "SHUTDOWN_COMMAND", fake_shutdown))
                stack.enter_context(mock.patch.object(
                    self.module, "acquire_lock", return_value=fake_lock))
                stack.enter_context(mock.patch.object(
                    self.module, "wait_for_runner",
                    side_effect=lambda *args: events.append("runner_exited")))
                stack.enter_context(mock.patch.object(
                    self.module, "wait_for_marker",
                    side_effect=lambda *args: (
                        events.append("marker_confirmed") or marker)))
                stack.enter_context(mock.patch.object(
                    self.module, "wait_for_no_training_processes",
                    side_effect=lambda *args: events.append("no_training")))
                stack.enter_context(mock.patch.object(
                    self.module, "write_audit",
                    side_effect=lambda *args: (
                        events.append("audit_written") or
                        output_dir / "SHUTDOWN_REQUESTED")))
                stack.enter_context(mock.patch.object(
                    self.module.os, "sync",
                    side_effect=lambda: events.append("synced")))
                shutdown_run = stack.enter_context(mock.patch.object(
                    self.module.subprocess, "run",
                    side_effect=lambda *args, **kwargs: (
                        events.append("shutdown_called") or
                        subprocess.CompletedProcess(args[0], 0))))
                stack.enter_context(mock.patch.object(sys, "argv", [
                    "shutdown_after_run.py",
                    "--pid", "1234",
                    "--expected-command", "relay.py",
                    "--output-dir", str(output_dir),
                    "--poweroff",
                ]))
                result = self.module.main()

            self.assertEqual(result, 0)
            self.assertEqual(events, [
                "runner_exited",
                "marker_confirmed",
                "no_training",
                "audit_written",
                "synced",
                "shutdown_called",
            ])
            shutdown_run.assert_called_once_with(
                ["/bin/bash", "-lc", str(fake_shutdown)],
                check=False,
                timeout=30,
            )
            fake_lock.close.assert_called_once_with()

    def test_poweroff_is_not_called_when_training_guard_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "outputs"
            output_dir = root / "relay"
            output_dir.mkdir(parents=True)
            marker = output_dir / "DONE"
            marker.write_text("done\n", encoding="utf-8")
            fake_shutdown = Path(temporary) / "shutdown"
            fake_shutdown.write_text("#!/bin/sh\n", encoding="utf-8")

            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(
                    self.module, "OUTPUT_ROOT", root.resolve()))
                stack.enter_context(mock.patch.object(
                    self.module, "SHUTDOWN_COMMAND", fake_shutdown))
                stack.enter_context(mock.patch.object(
                    self.module, "acquire_lock", mock.Mock()))
                stack.enter_context(mock.patch.object(
                    self.module, "wait_for_runner"))
                stack.enter_context(mock.patch.object(
                    self.module, "wait_for_marker", return_value=marker))
                stack.enter_context(mock.patch.object(
                    self.module, "wait_for_no_training_processes",
                    side_effect=RuntimeError("training still active")))
                sync = stack.enter_context(mock.patch.object(
                    self.module.os, "sync"))
                shutdown_run = stack.enter_context(mock.patch.object(
                    self.module.subprocess, "run"))
                stack.enter_context(mock.patch.object(sys, "argv", [
                    "shutdown_after_run.py",
                    "--pid", "1234",
                    "--expected-command", "relay.py",
                    "--output-dir", str(output_dir),
                    "--poweroff",
                ]))
                with self.assertRaisesRegex(
                        RuntimeError, "training still active"):
                    self.module.main()

            sync.assert_not_called()
            shutdown_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
