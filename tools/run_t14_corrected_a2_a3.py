#!/usr/bin/env python3
"""Run corrected A2 and launch corrected A3 only if A2 clears A1."""

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path("/root/autodl-tmp/outputs/HTL-ReID")
CHAIN_NAME = "E053_E054_corrected_A2_A3"
A1_MAP = 68.10
A1_RANK1 = 69.02
ROWS = (
    ("E053", "A2-FACR-RESIDUAL",
     "configs/RGBNT201/ablations/t14_chain/a2_facr_residual.yml",
     "E053_A2_facr_residual_RGBNT201_seed1111"),
    ("E054", "A3-PART-RESIDUAL",
     "configs/RGBNT201/ablations/t14_chain/a3_part_residual.yml",
     "E054_A3_part_residual_RGBNT201_seed1111"),
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


def read_result(output_dir, experiment, row, config, returncode, commit):
    result_path = output_dir / "run_result.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    return {
        "experiment": experiment,
        "row": row,
        "status": "runner_failed",
        "returncode": returncode,
        "config": config,
        "output_dir": str(output_dir),
        "commit": commit,
    }


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
            completed = subprocess.run(command, cwd=args.repo_root, check=False)
            if completed.returncode != 0:
                return completed.returncode
        print(json.dumps({
            "a3_gate": {
                "a2_status": "completed",
                "minimum_mAP": A1_MAP,
                "minimum_Rank1": A1_RANK1,
            }
        }, indent=2, ensure_ascii=False))
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
    a2_completed = subprocess.run(
        commands[0], cwd=args.repo_root, check=False)
    a2_result = read_result(
        output_dirs[0], *ROWS[0][:3], a2_completed.returncode, commit)
    a2_passed = (
        a2_result.get("status") == "completed" and
        float(a2_result.get("mAP", float("-inf"))) >= A1_MAP and
        float(a2_result.get("Rank1", float("-inf"))) >= A1_RANK1
    )
    a2_result["a3_gate_passed"] = a2_passed
    a2_result["a3_gate_minimum_mAP"] = A1_MAP
    a2_result["a3_gate_minimum_Rank1"] = A1_RANK1
    results.append(a2_result)
    chain_dir.joinpath("summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    if not a2_passed:
        chain_dir.joinpath("RUNNING").unlink(missing_ok=True)
        chain_dir.joinpath("STOPPED_AFTER_A2").write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        return 2

    a3_completed = subprocess.run(
        commands[1], cwd=args.repo_root, check=False)
    a3_result = read_result(
        output_dirs[1], *ROWS[1][:3], a3_completed.returncode, commit)
    results.append(a3_result)
    chain_dir.joinpath("summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    chain_dir.joinpath("RUNNING").unlink(missing_ok=True)
    marker = "DONE" if a3_result.get("status") == "completed" else "FAILED"
    chain_dir.joinpath(marker).write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return 0 if marker == "DONE" else 1


if __name__ == "__main__":
    sys.exit(main())
