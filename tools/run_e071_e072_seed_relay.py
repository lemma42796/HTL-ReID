#!/usr/bin/env python3
"""Run the unchanged E060 model with seed 2222, then seed 3333 if needed.

Each child is executed through run_rgbnt201_fusion.py, which enforces the
30-minute wall-clock limit for every training process. This coordinator writes
a chain-level DONE or FAILED marker for shutdown_after_run.py; it never powers
off the host itself.
"""

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path("/root/autodl-tmp/outputs/HTL-ReID")
CHAIN_NAME = "E071_E072_seed_relay"
TARGET_MAP = 70.0
CONFIG = "configs/RGBNT201/fusion/a3_isolated_clean.yml"
BASE_CONFIG = "configs/RGBNT201/paper/base.yml"

STEPS = (
    {
        "experiment": "E071",
        "row": "A3-I-S2222-R1",
        "seed": 2222,
        "output_name": "E071_A3_isolated_seed2222_recovery_RGBNT201",
    },
    {
        "experiment": "E072",
        "row": "A3-I-S3333",
        "seed": 3333,
        "output_name": "E072_A3_isolated_seed3333_RGBNT201",
    },
)


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def child_command(python, spec, dry_run=False):
    command = [
        python,
        "tools/run_rgbnt201_fusion.py",
        "--single-experiment", spec["experiment"],
        "--single-row", spec["row"],
        "--single-config", CONFIG,
        "--single-output-name", spec["output_name"],
        "--base-config", BASE_CONFIG,
        "--dataset", "RGBNT201",
        "--input-size", "256", "128",
        "--seed", str(spec["seed"]),
        "--eval-period", "1",
        "--expected-train-epochs", "50",
        "--expected-max-epochs", "50",
        "--expected-base-lr", "0.0001",
        "--expected-batch-size", "64",
        "--expected-backbone-lr-factor", "0.1",
        "--expected-warmup-iters", "10",
        "--expected-resume-path", "",
        "--expected-strict-determinism", "0",
    ]
    if dry_run:
        command.append("--dry-run")
    return command


def read_result(output_dir, spec, returncode, commit):
    result_path = output_dir / "run_result.json"
    if result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))
    return {
        "experiment": spec["experiment"],
        "row": spec["row"],
        "seed": spec["seed"],
        "status": "runner_failed",
        "returncode": returncode,
        "config": CONFIG,
        "output_dir": str(output_dir),
        "commit": commit,
    }


def completed_map(result):
    if result.get("status") != "completed":
        return None
    try:
        return float(result["mAP"])
    except (KeyError, TypeError, ValueError):
        return None


def run_one(args, spec, commit):
    command = child_command(args.python, spec)
    completed = subprocess.run(command, cwd=args.repo_root, check=False)
    output_dir = args.output_root / spec["output_name"]
    return read_result(output_dir, spec, completed.returncode, commit)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="/root/miniconda3/bin/python")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.output_root = args.output_root.resolve()

    if args.dry_run:
        for spec in STEPS:
            completed = subprocess.run(
                child_command(args.python, spec, dry_run=True),
                cwd=args.repo_root,
                check=False,
            )
            if completed.returncode != 0:
                return completed.returncode
        print(json.dumps({
            "target_mAP_strictly_greater_than": TARGET_MAP,
            "first": STEPS[0],
            "fallback_if_first_fails_or_misses_target": STEPS[1],
            "re_ranking": "no",
            "tta": False,
        }, indent=2, ensure_ascii=False))
        return 0

    chain_dir = args.output_root / CHAIN_NAME
    output_dirs = [
        args.output_root / spec["output_name"] for spec in STEPS
    ]
    conflicts = [
        path for path in (chain_dir, *output_dirs) if path.exists()
    ]
    if conflicts:
        raise FileExistsError(
            "refusing to overwrite: {}".format(
                ", ".join(str(path) for path in conflicts)))

    chain_dir.mkdir(parents=True)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=args.repo_root,
        text=True,
    ).strip()
    plan = {
        "commit": commit,
        "target_mAP_strictly_greater_than": TARGET_MAP,
        "config": CONFIG,
        "steps": STEPS,
        "rule": (
            "run E071; run E072 only if E071 fails or mAP <= 70.0; "
            "write a terminal chain marker for the separate shutdown watcher"
        ),
    }
    write_json(chain_dir / "plan.json", plan)
    (chain_dir / "RUNNING").write_text(
        dt.datetime.now().astimezone().isoformat() + "\n",
        encoding="utf-8",
    )

    results = []
    first = run_one(args, STEPS[0], commit)
    results.append(first)
    write_json(chain_dir / "summary.json", results)
    first_map = completed_map(first)

    if first_map is not None and first_map > TARGET_MAP:
        decision = {
            "target_achieved": True,
            "selected_experiment": "E071",
            "selected_mAP": first_map,
            "e072_skipped": True,
        }
    else:
        second = run_one(args, STEPS[1], commit)
        results.append(second)
        write_json(chain_dir / "summary.json", results)
        second_map = completed_map(second)
        valid = [
            (result["experiment"], value)
            for result, value in (
                (first, first_map), (second, second_map))
            if value is not None
        ]
        selected = max(valid, key=lambda item: item[1]) if valid else None
        decision = {
            "target_achieved": bool(
                selected is not None and selected[1] > TARGET_MAP),
            "selected_experiment": selected[0] if selected else None,
            "selected_mAP": selected[1] if selected else None,
            "e072_skipped": False,
        }

    write_json(chain_dir / "decision.json", decision)
    (chain_dir / "RUNNING").unlink(missing_ok=True)
    failed = [
        result for result in results
        if result.get("status") != "completed"
    ]
    marker = "DONE" if not failed else "FAILED"
    write_json(chain_dir / marker, {
        "decision": decision,
        "results": results,
    })
    return 0 if not failed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        chain_dir = OUTPUT_ROOT / CHAIN_NAME
        if chain_dir.is_dir():
            (chain_dir / "RUNNING").unlink(missing_ok=True)
            write_json(chain_dir / "FAILED", {
                "error": str(exc),
                "time": dt.datetime.now().astimezone().isoformat(),
            })
        raise
