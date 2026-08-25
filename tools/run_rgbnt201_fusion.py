#!/usr/bin/env python3
"""Run T1, T2, and T3 fusion rows sequentially with independent 30m caps."""

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
ROWS = (
    ("E006", "T1", "configs/RGBNT201/fusion/t1_tpm.yml"),
    ("E007", "T2", "configs/RGBNT201/fusion/t2_adaptive_routing.yml"),
    ("E008", "T3", "configs/RGBNT201/fusion/t3_m2_facr.yml"),
)
OUTPUT_NAMES = {
    "T1": "E006_T1_tpm_seed1111",
    "T2": "E007_T2_adaptive_routing_seed1111",
    "T3": "E008_T3_m2_facr_seed1111",
}
TIME_LIMIT = "30m"
TRAIN_EPOCHS = 20
SEED = 1111
BATCH_SIZE = 40
EVAL_PERIOD = 1

EPOCH_PATTERN = re.compile(r"Validation Results - Epoch:\s+(\d+)")
MAP_PATTERN = re.compile(r"mAP:\s+([0-9.]+)%")
RANK_PATTERNS = {
    "Rank1": re.compile(r"Rank-1\s*:([^%]+)%"),
    "Rank5": re.compile(r"Rank-5\s*:([^%]+)%"),
    "Rank10": re.compile(r"Rank-10\s*:([^%]+)%"),
}


def resolve_config(row_config, output_dir):
    cfg = default_cfg.clone()
    cfg.merge_from_file(BASE_CONFIG)
    cfg.merge_from_file(row_config)
    cfg.OUTPUT_DIR = str(output_dir)
    cfg.SOLVER.EVAL_PERIOD = EVAL_PERIOD
    if int(cfg.SOLVER.TRAIN_EPOCHS) != TRAIN_EPOCHS:
        raise ValueError("all fusion rows must train for exactly 20 epochs")
    if int(cfg.SOLVER.SEED) != SEED:
        raise ValueError("all fusion rows must use seed 1111")
    if int(cfg.SOLVER.IMS_PER_BATCH) != BATCH_SIZE:
        raise ValueError("all fusion rows must use batch size 40")
    if str(cfg.TEST.RE_RANKING).lower() != "no":
        raise ValueError("fusion rows must disable re-ranking")
    if cfg.MODEL.PRETRAIN_CHOICE != "imagenet" or cfg.MODEL.RESUME_PATH:
        raise ValueError("each row must start independently from ImageNet weights")
    return cfg


def parse_best(log_path):
    validations = []
    current = None
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            epoch_match = EPOCH_PATTERN.search(line)
            if epoch_match:
                current = {"best_epoch": int(epoch_match.group(1))}
                validations.append(current)
                continue
            if current is None:
                continue
            map_match = MAP_PATTERN.search(line)
            if map_match and "Best Multi-Modal" not in line:
                current["mAP"] = float(map_match.group(1))
            for key, pattern in RANK_PATTERNS.items():
                rank_match = pattern.search(line)
                if rank_match and "Best Multi-Modal" not in line:
                    current[key] = float(rank_match.group(1).strip())
    complete = [item for item in validations if "mAP" in item]
    if not complete:
        return {"best_epoch": "", "mAP": "", "Rank1": "", "Rank5": "", "Rank10": ""}
    return max(complete, key=lambda item: item["mAP"])


def write_summary(path, results):
    fields = (
        "experiment", "row", "status", "returncode", "elapsed_seconds",
        "best_epoch", "mAP", "Rank1", "Rank5", "Rank10", "output_dir",
        "config", "commit",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


def run_row(args, timeout_bin, commit, experiment, row, row_config):
    output_dir = args.output_root / OUTPUT_NAMES[row]
    output_dir.mkdir(parents=True)
    cfg = resolve_config(row_config, output_dir)
    output_dir.joinpath("resolved_config.yml").write_text(cfg.dump(), encoding="utf-8")
    output_dir.joinpath("commit.txt").write_text(commit + "\n", encoding="utf-8")
    command = [
        timeout_bin, "--signal=TERM", "--kill-after=10s", TIME_LIMIT,
        args.python, "train_net.py",
        "--config_file", BASE_CONFIG,
        "--config_file", row_config,
        "OUTPUT_DIR", str(output_dir),
        "SOLVER.EVAL_PERIOD", str(EVAL_PERIOD),
    ]
    output_dir.joinpath("command.txt").write_text(
        " ".join(command) + "\n", encoding="utf-8")
    output_dir.joinpath("RUNNING").write_text(
        dt.datetime.now().astimezone().isoformat() + "\n", encoding="utf-8")

    log_path = output_dir / "stdout.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(" ".join(command) + "\n\n")
        log_handle.flush()
        completed = subprocess.run(
            command, cwd=args.repo_root, stdout=log_handle,
            stderr=subprocess.STDOUT, check=False)
    elapsed = round(time.monotonic() - started, 1)
    output_dir.joinpath("RUNNING").unlink(missing_ok=True)
    status = "completed" if completed.returncode == 0 else (
        "timeout" if completed.returncode == 124 else "failed")
    result = {
        "experiment": experiment,
        "row": row,
        "status": status,
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        **parse_best(log_path),
        "output_dir": str(output_dir),
        "config": row_config,
        "commit": commit,
    }
    output_dir.joinpath("run_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    marker = "DONE" if completed.returncode == 0 else "FAILED"
    output_dir.joinpath(marker).write_text(
        json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="/root/miniconda3/bin/python")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("/root/autodl-tmp/outputs/HTL-ReID"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.output_root = args.output_root.resolve()

    plan = []
    for experiment, row, row_config in ROWS:
        output_dir = args.output_root / OUTPUT_NAMES[row]
        resolve_config(row_config, output_dir)
        plan.append({
            "experiment": experiment,
            "row": row,
            "base_config": BASE_CONFIG,
            "row_config": row_config,
            "output_dir": str(output_dir),
            "dataset": "RGBNT201",
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "epochs": TRAIN_EPOCHS,
            "eval_period": EVAL_PERIOD,
            "re_ranking": "no",
            "time_limit_per_row": TIME_LIMIT,
        })
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0

    timeout_bin = shutil.which("timeout")
    if timeout_bin is None:
        raise RuntimeError("GNU timeout is required")
    args.output_root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.output_root).free < 3 * 1024 ** 3:
        raise RuntimeError("less than 3 GiB free under output root")
    for item in plan:
        if Path(item["output_dir"]).exists():
            raise FileExistsError("refusing to overwrite {}".format(item["output_dir"]))

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.repo_root, text=True).strip()
    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_root / "fusion_E006-E008_{}".format(stamp)
    run_dir.mkdir(parents=True)
    run_dir.joinpath("plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    run_dir.joinpath("commit.txt").write_text(commit + "\n", encoding="utf-8")

    results = []
    for experiment, row, row_config in ROWS:
        results.append(run_row(
            args, timeout_bin, commit, experiment, row, row_config))
        write_summary(run_dir / "summary.csv", results)
    failed = [result for result in results if result["status"] != "completed"]
    marker = "DONE" if not failed else "FAILED"
    run_dir.joinpath(marker).write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
