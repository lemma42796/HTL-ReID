#!/usr/bin/env python3
"""Shut down an AutoDL instance after one validated aggregate run finishes."""

import argparse
import datetime as dt
import os
import subprocess
import sys
import time
from pathlib import Path


OUTPUT_ROOT = Path("/root/autodl-tmp/outputs/HTL-ReID").resolve()
SHUTDOWN_COMMAND = Path("/usr/bin/shutdown")


def timestamp():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def log(message):
    print("{} {}".format(timestamp(), message), flush=True)


def command_line(pid):
    try:
        raw = Path("/proc/{}/cmdline".format(pid)).read_bytes()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()


def active_train_pids():
    pids = []
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        cmd = command_line(int(proc_dir.name))
        if "train_net.py" in cmd:
            pids.append(int(proc_dir.name))
    return sorted(pids)


def validate_run_dir(path):
    resolved = path.resolve()
    if resolved.parent != OUTPUT_ROOT:
        raise ValueError("run directory must be a direct child of {}".format(OUTPUT_ROOT))
    if not resolved.name.startswith("fusion_E006-E008_"):
        raise ValueError("refusing unrelated run directory {}".format(resolved))
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    if not resolved.joinpath("plan.json").is_file():
        raise FileNotFoundError("missing plan.json under {}".format(resolved))
    return resolved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--runner-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    run_dir = validate_run_dir(args.run_dir)
    if args.poll_seconds < 5:
        raise ValueError("poll interval must be at least 5 seconds")
    initial_cmd = command_line(args.runner_pid)
    if "run_rgbnt201_fusion.py" not in initial_cmd:
        raise RuntimeError(
            "PID {} is not the fusion runner: {}".format(args.runner_pid, initial_cmd))
    if not SHUTDOWN_COMMAND.is_file():
        raise FileNotFoundError(SHUTDOWN_COMMAND)

    log("monitoring runner PID {} for {}".format(args.runner_pid, run_dir))
    while True:
        marker = None
        for name in ("DONE", "FAILED"):
            candidate = run_dir / name
            if candidate.is_file():
                marker = candidate
                break

        if marker is not None:
            train_pids = active_train_pids()
            if train_pids:
                log("{} exists but train processes remain {}; waiting".format(
                    marker.name, train_pids))
                time.sleep(args.poll_seconds)
                continue
            request = run_dir / "SHUTDOWN_REQUESTED"
            request.write_text(
                "{} marker={} runner_pid={}\n".format(
                    timestamp(), marker.name, args.runner_pid),
                encoding="utf-8",
            )
            log("{} confirmed and no train process remains; requesting AutoDL shutdown".format(
                marker.name))
            os.sync()
            completed = subprocess.run(
                [str(SHUTDOWN_COMMAND)], check=False, timeout=30)
            if completed.returncode != 0:
                log("shutdown command failed with return code {}".format(
                    completed.returncode))
                return 3
            return 0

        current_cmd = command_line(args.runner_pid)
        if "run_rgbnt201_fusion.py" not in current_cmd:
            log("runner exited without DONE/FAILED marker; refusing shutdown")
            run_dir.joinpath("SHUTDOWN_REFUSED").write_text(
                "{} runner exited without aggregate marker\n".format(timestamp()),
                encoding="utf-8",
            )
            return 2
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
