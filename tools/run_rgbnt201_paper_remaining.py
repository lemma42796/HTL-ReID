#!/usr/bin/env python3
"""Run the remaining formal RGBNT201 paper rows (M1-M3) sequentially.

Each row starts independently from the configured ImageNet pretrained weight,
has its own output directory, and is protected by a 30-minute wall-clock cap.
The runner writes a resolved config, command, log, result JSON, and an aggregate
CSV summary. A failed row is recorded and does not prevent later independent
rows from running.
"""

import argparse
import csv
import datetime as dt
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import cfg as default_cfg


BASE_CONFIG = "configs/RGBNT201/paper/base.yml"
ROWS = [
    ("E002", "M1", "m1", "configs/RGBNT201/paper/m1.yml"),
    ("E003", "M2", "m2", "configs/RGBNT201/paper/m2.yml"),
    ("E004", "M3", "m3", "configs/RGBNT201/paper/m3.yml"),
]
TIME_LIMIT = "30m"
EXPECTED_TRAIN_EPOCHS = 50
EXPECTED_SEED = 1111

EPOCH_PATTERN = re.compile(r"Validation Results - Epoch:\s+(\d+)")
MAP_PATTERN = re.compile(r"mAP:\s+([0-9.]+)%")
RANK_PATTERNS = {
    "rank1": re.compile(r"Rank-1\s*:([^%]+)%"),
    "rank5": re.compile(r"Rank-5\s*:([^%]+)%"),
    "rank10": re.compile(r"Rank-10\s*:([^%]+)%"),
}


def resolved_cfg(row_config, output_dir):
    cfg = default_cfg.clone()
    cfg.merge_from_file(BASE_CONFIG)
    cfg.merge_from_file(row_config)
    cfg.OUTPUT_DIR = str(output_dir)
    if int(cfg.SOLVER.TRAIN_EPOCHS) != EXPECTED_TRAIN_EPOCHS:
        raise ValueError("paper rows must train for exactly 50 epochs")
    if int(cfg.SOLVER.SEED) != EXPECTED_SEED:
        raise ValueError("paper rows must use seed 1111")
    if str(cfg.TEST.RE_RANKING).lower() != "no":
        raise ValueError("main paper rows must disable re-ranking")
    if cfg.MODEL.PRETRAIN_CHOICE != "imagenet" or cfg.MODEL.RESUME_PATH:
        raise ValueError("each row must start independently from ImageNet weights")
    return cfg


def parse_best_metrics(log_path):
    validations = []
    current = None
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            epoch_match = EPOCH_PATTERN.search(line)
            if epoch_match:
                current = {"epoch": int(epoch_match.group(1))}
                validations.append(current)
                continue
            if current is None:
                continue
            map_match = MAP_PATTERN.search(line)
            if map_match and "Best Multi-Modal" not in line:
                current["map"] = float(map_match.group(1))
            for key, pattern in RANK_PATTERNS.items():
                match = pattern.search(line)
                if match and "Best Multi-Modal" not in line:
                    current[key] = float(match.group(1).strip())

    complete = [item for item in validations if "map" in item]
    if not complete:
        return {"best_epoch": "", "mAP": "", "Rank1": "", "Rank5": "", "Rank10": ""}
    best = max(complete, key=lambda item: item["map"])
    return {
        "best_epoch": best["epoch"],
        "mAP": best.get("map", ""),
        "Rank1": best.get("rank1", ""),
        "Rank5": best.get("rank5", ""),
        "Rank10": best.get("rank10", ""),
    }


def write_csv(path, results):
    fields = [
        "experiment", "paper_row", "status", "returncode", "elapsed_seconds",
        "best_epoch", "mAP", "Rank1", "Rank5", "Rank10", "output_dir", "config",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


def run_row(args, timeout_bin, commit, experiment, paper_row, name, row_config):
    output_dir = args.output_root / f"{experiment}_{paper_row}_seed{EXPECTED_SEED}"
    output_dir.mkdir(parents=True)
    cfg = resolved_cfg(row_config, output_dir)
    output_dir.joinpath("resolved_config.yml").write_text(cfg.dump(), encoding="utf-8")
    output_dir.joinpath("commit.txt").write_text(commit + "\n", encoding="utf-8")

    train_cmd = [
        args.python,
        "train_net.py",
        "--config_file", BASE_CONFIG,
        "--config_file", row_config,
        "OUTPUT_DIR", str(output_dir),
    ]
    capped_cmd = [
        timeout_bin, "--signal=TERM", "--kill-after=10s", TIME_LIMIT,
        *train_cmd,
    ]
    output_dir.joinpath("command.txt").write_text(
        " ".join(capped_cmd) + "\n", encoding="utf-8")
    output_dir.joinpath("RUNNING").write_text(
        dt.datetime.now().astimezone().isoformat() + "\n", encoding="utf-8")

    log_path = output_dir / "stdout.log"
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(" ".join(capped_cmd) + "\n\n")
        log_handle.flush()
        completed = subprocess.run(
            capped_cmd,
            cwd=args.repo_root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = round(time.monotonic() - start, 1)
    output_dir.joinpath("RUNNING").unlink(missing_ok=True)
    metrics = parse_best_metrics(log_path)
    status = "completed" if completed.returncode == 0 else (
        "timeout" if completed.returncode == 124 else "failed")
    result = {
        "experiment": experiment,
        "paper_row": paper_row,
        "status": status,
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        **metrics,
        "output_dir": str(output_dir),
        "config": row_config,
    }
    output_dir.joinpath("run_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output_dir.joinpath("DONE" if completed.returncode == 0 else "FAILED").write_text(
        json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def build_plan(args):
    return [
        {
            "experiment": experiment,
            "paper_row": paper_row,
            "base_config": BASE_CONFIG,
            "row_config": row_config,
            "output_dir": str(args.output_root / f"{experiment}_{paper_row}_seed{EXPECTED_SEED}"),
            "epochs": EXPECTED_TRAIN_EPOCHS,
            "seed": EXPECTED_SEED,
            "time_limit": TIME_LIMIT,
        }
        for experiment, paper_row, _, row_config in ROWS
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="/root/miniconda3/bin/python")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("/root/autodl-tmp/outputs/HTL-ReID"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.output_root = args.output_root.resolve()

    plan = build_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0

    timeout_bin = shutil.which("timeout")
    if timeout_bin is None:
        raise RuntimeError("GNU timeout is required to enforce the 30-minute cap")
    usage = shutil.disk_usage(args.output_root)
    if usage.free < 2 * 1024 ** 3:
        raise RuntimeError("less than 2 GiB free under output root")
    for item in plan:
        if Path(item["output_dir"]).exists():
            raise FileExistsError("refusing to overwrite {}".format(item["output_dir"]))

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.repo_root, text=True).strip()
    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_root / f"paper_remaining_E002-E004_{stamp}"
    run_dir.mkdir(parents=True)
    run_dir.joinpath("plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    run_dir.joinpath("commit.txt").write_text(commit + "\n", encoding="utf-8")

    results = []
    for experiment, paper_row, name, row_config in ROWS:
        result = run_row(
            args, timeout_bin, commit, experiment, paper_row, name, row_config)
        results.append(result)
        write_csv(run_dir / "summary.csv", results)

    failed = [item for item in results if item["status"] != "completed"]
    run_dir.joinpath("DONE" if not failed else "FAILED").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
