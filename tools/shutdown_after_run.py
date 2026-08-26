#!/usr/bin/env python3
"""Safely power off a training host after one registered run finishes.

The watcher refuses to shut down unless all of the following are true:

1. The watched PID initially belongs to the expected runner command (unless it
   has already exited).
2. The watched process exits without its PID being reused by another command.
3. The registered output directory contains a DONE or FAILED marker.
4. No train_net.py process remains on the host.
5. The caller explicitly passes --poweroff (or uses --dry-run for validation).

A non-blocking lock in the output directory prevents duplicate watchers.
"""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


OUTPUT_ROOT = Path("/root/autodl-tmp/outputs/HTL-ReID").resolve()
SHUTDOWN_COMMAND = Path("/usr/bin/shutdown")


def log(message):
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print("{} {}".format(stamp, message), flush=True)


def read_command(pid):
    try:
        payload = Path("/proc/{}/cmdline".format(pid)).read_bytes()
    except (FileNotFoundError, ProcessLookupError):
        return None
    except PermissionError as exc:
        raise RuntimeError(
            "cannot inspect watched PID {}: {}".format(pid, exc)) from exc
    return payload.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()


def active_train_pids():
    pids = []
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        command = read_command(int(proc_dir.name))
        if command is not None and "train_net.py" in command:
            pids.append(int(proc_dir.name))
    return sorted(pids)


def wait_for_runner(pid, expected_command, poll_seconds):
    command = read_command(pid)
    if command is None:
        log("watched PID {} has already exited".format(pid))
        return
    if expected_command not in command:
        raise RuntimeError(
            "PID {} does not match expected command {!r}: {}".format(
                pid, expected_command, command))

    log("watching PID {}: {}".format(pid, command))
    while True:
        time.sleep(poll_seconds)
        command = read_command(pid)
        if command is None:
            log("watched PID {} exited".format(pid))
            return
        if expected_command not in command:
            raise RuntimeError(
                "PID {} was reused or changed command before completion: {}".format(
                    pid, command))


def wait_for_marker(output_dir, grace_seconds, poll_seconds):
    deadline = time.monotonic() + grace_seconds
    while True:
        for name in ("DONE", "FAILED"):
            marker = output_dir / name
            if marker.is_file():
                log("confirmed completion marker {}".format(marker))
                return marker
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "runner exited but no DONE/FAILED marker appeared within {:.1f}s".format(
                    grace_seconds))
        time.sleep(min(poll_seconds, max(deadline - time.monotonic(), 0.05)))


def wait_for_no_training_processes(grace_seconds, poll_seconds):
    deadline = time.monotonic() + grace_seconds
    while True:
        pids = active_train_pids()
        if not pids:
            log("confirmed no train_net.py process remains")
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "training processes still active after {:.1f}s: {}".format(
                    grace_seconds, pids))
        log("training processes still active {}; waiting".format(pids))
        time.sleep(min(poll_seconds, max(deadline - time.monotonic(), 0.05)))


def acquire_lock(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".shutdown_after_run.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(
            "another shutdown watcher already holds {}".format(lock_path)) from exc
    return handle


def write_audit(output_dir, marker, mode):
    path = output_dir / (
        "SHUTDOWN_DRY_RUN_OK" if mode == "dry-run" else "SHUTDOWN_REQUESTED")
    payload = {
        "time": datetime.now().astimezone().isoformat(),
        "completion_marker": str(marker),
        "mode": mode,
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--expected-command", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--marker-grace-seconds", type=float, default=120.0)
    parser.add_argument("--process-grace-seconds", type=float, default=120.0)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--poweroff", action="store_true")
    args = parser.parse_args()

    if args.pid <= 1:
        parser.error("--pid must identify a non-system process")
    if args.poll_seconds <= 0.0:
        parser.error("--poll-seconds must be positive")
    if args.marker_grace_seconds < 0.0:
        parser.error("--marker-grace-seconds must be non-negative")
    if args.process_grace_seconds < 0.0:
        parser.error("--process-grace-seconds must be non-negative")

    output_dir = args.output_dir.resolve()
    if args.poweroff:
        if output_dir.parent != OUTPUT_ROOT:
            raise RuntimeError(
                "poweroff output directory must be a direct child of {}".format(
                    OUTPUT_ROOT))
        if not output_dir.is_dir():
            raise FileNotFoundError(output_dir)
        if not SHUTDOWN_COMMAND.is_file():
            raise FileNotFoundError(SHUTDOWN_COMMAND)
    lock_handle = acquire_lock(output_dir)

    wait_for_runner(args.pid, args.expected_command, args.poll_seconds)
    marker = wait_for_marker(
        output_dir, args.marker_grace_seconds, args.poll_seconds)

    if args.dry_run:
        audit = write_audit(output_dir, marker, "dry-run")
        log("dry-run complete; would verify no train_net.py process remains")
        log("dry-run complete; would execute: /bin/bash -lc /usr/bin/shutdown")
        log("wrote audit marker {}".format(audit))
        return 0

    wait_for_no_training_processes(
        args.process_grace_seconds, args.poll_seconds)
    audit = write_audit(output_dir, marker, "poweroff")
    log("wrote audit marker {}".format(audit))
    log("syncing filesystems before poweroff")
    os.sync()
    log("executing /bin/bash -lc {}".format(SHUTDOWN_COMMAND))
    completed = subprocess.run(
        ["/bin/bash", "-lc", str(SHUTDOWN_COMMAND)],
        check=False,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "shutdown command failed with return code {}".format(
                completed.returncode))
    # Keep the lock file descriptor alive until the shutdown command returns.
    lock_handle.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log("refusing to power off: {}".format(exc))
        raise
