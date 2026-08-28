#!/usr/bin/env python3
"""Run the clean RGBNT201 A0-A3 cumulative ablation chain."""

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path("/root/autodl-tmp/outputs/HTL-ReID")
CHAIN_NAME = "E049_E052_T14_cumulative_chain"
ROWS = (
    ("E049", "A0-BACKBONE", "configs/RGBNT201/ablations/t14_chain/a0_backbone.yml",
     "E049_A0_backbone_RGBNT201_seed1111"),
    ("E050", "A1-SFTS", "configs/RGBNT201/ablations/t14_chain/a1_sfts.yml",
     "E050_A1_sfts_RGBNT201_seed1111"),
    ("E051", "A2-FACR", "configs/RGBNT201/ablations/t14_chain/a2_facr.yml",
     "E051_A2_facr_RGBNT201_seed1111"),
    ("E052", "A3-PART", "configs/RGBNT201/ablations/t14_chain/a3_part.yml",
     "E052_A3_part_RGBNT201_seed1111"),
)


def child_command(python, experiment, row, config, output_name, dry_run=False):
    command = [
        python,
        "tools/run_rgbnt201_fusion.py",
        "--single-experiment", experiment,
        "--single-row", row,
        "--single-config", config,
        "--single-output-name", output_name,
        "--seed", "1111",
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="/root/miniconda3/bin/python")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.output_root = args.output_root.resolve()

    commands = [
        child_command(args.python, experiment, row, config, output_name,
                      dry_run=args.dry_run)
        for experiment, row, config, output_name in ROWS
    ]
    if args.dry_run:
        for command in commands:
            completed = subprocess.run(
                command, cwd=args.repo_root, check=False)
            if completed.returncode != 0:
                return completed.returncode
        return 0

    chain_dir = args.output_root / CHAIN_NAME
    output_dirs = [args.output_root / item[3] for item in ROWS]
    conflicts = [path for path in [chain_dir, *output_dirs] if path.exists()]
    if conflicts:
        raise FileExistsError(
            "refusing to overwrite: {}".format(
                ", ".join(str(path) for path in conflicts)))

    chain_dir.mkdir(parents=True)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.repo_root, text=True).strip()
    plan = []
    for (experiment, row, config, output_name), command in zip(ROWS, commands):
        plan.append({
            "experiment": experiment,
            "row": row,
            "config": config,
            "output_dir": str(args.output_root / output_name),
            "command": command,
        })
    chain_dir.joinpath("plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    chain_dir.joinpath("commit.txt").write_text(commit + "\n", encoding="utf-8")
    chain_dir.joinpath("RUNNING").write_text(
        dt.datetime.now().astimezone().isoformat() + "\n", encoding="utf-8")

    results = []
    for (experiment, row, config, output_name), command in zip(ROWS, commands):
        completed = subprocess.run(command, cwd=args.repo_root, check=False)
        result_path = args.output_root / output_name / "run_result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            result = {
                "experiment": experiment,
                "row": row,
                "status": "runner_failed",
                "returncode": completed.returncode,
                "config": config,
                "output_dir": str(args.output_root / output_name),
                "commit": commit,
            }
        results.append(result)
        chain_dir.joinpath("summary.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")

    chain_dir.joinpath("RUNNING").unlink(missing_ok=True)
    failed = [item for item in results if item.get("status") != "completed"]
    marker = "DONE" if not failed else "FAILED"
    chain_dir.joinpath(marker).write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
