#!/usr/bin/env python3
"""Run the T1-T3 batch or one registered RGBNT201 fusion experiment."""

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
TRAIN_EPOCHS = 50
DEFAULT_SEED = 1111
BATCH_SIZE = 128
EVAL_PERIOD = 1
INPUT_SIZE = (256, 128)
SUPPORTED_BACKBONE_LR_FACTORS = (0.2, 0.8)

EPOCH_PATTERN = re.compile(r"Validation Results - Epoch:\s+(\d+)")
MAP_PATTERN = re.compile(r"mAP:\s+([0-9.]+)%")
RANK_PATTERNS = {
    "Rank1": re.compile(r"Rank-1\s*:([^%]+)%"),
    "Rank5": re.compile(r"Rank-5\s*:([^%]+)%"),
    "Rank10": re.compile(r"Rank-10\s*:([^%]+)%"),
}


def resolve_config(row_config, output_dir, seed=DEFAULT_SEED,
                   expected_train_epochs=TRAIN_EPOCHS,
                   expected_base_lr=3.5e-4):
    cfg = default_cfg.clone()
    cfg.merge_from_file(BASE_CONFIG)
    cfg.merge_from_file(row_config)
    cfg.OUTPUT_DIR = str(output_dir)
    cfg.SOLVER.EVAL_PERIOD = EVAL_PERIOD
    cfg.SOLVER.SEED = int(seed)
    if int(cfg.SOLVER.TRAIN_EPOCHS) != int(expected_train_epochs):
        raise ValueError(
            "expected {} training epochs, got {}".format(
                expected_train_epochs, cfg.SOLVER.TRAIN_EPOCHS))
    if int(cfg.SOLVER.SEED) != int(seed):
        raise ValueError("fusion row seed override was not applied")
    if int(cfg.SOLVER.IMS_PER_BATCH) != BATCH_SIZE:
        raise ValueError("all RGBNT201 paper rows must use batch size 128")
    if tuple(cfg.INPUT.SIZE_TRAIN) != INPUT_SIZE:
        raise ValueError("all RGBNT201 fusion rows must train at 256x128")
    if tuple(cfg.INPUT.SIZE_TEST) != INPUT_SIZE:
        raise ValueError("all RGBNT201 fusion rows must test at 256x128")
    if cfg.SOLVER.OPTIMIZER_NAME != "Adam":
        raise ValueError("all RGBNT201 paper rows must use Adam")
    if abs(float(cfg.SOLVER.BASE_LR) - float(expected_base_lr)) > 1e-12:
        raise ValueError(
            "expected base LR {}, got {}".format(
                expected_base_lr, cfg.SOLVER.BASE_LR))
    backbone_lr_factor = float(cfg.SOLVER.BACKBONE_LR_FACTOR)
    if not any(abs(backbone_lr_factor - value) <= 1e-12
               for value in SUPPORTED_BACKBONE_LR_FACTORS):
        raise ValueError(
            "RGBNT201 runner supports backbone LR factors {}, got {}".format(
                SUPPORTED_BACKBONE_LR_FACTORS, backbone_lr_factor))
    if abs(float(cfg.SOLVER.WARMUP_FACTOR) - 0.1) > 1e-12:
        raise ValueError("all RGBNT201 paper rows must warm up from 0.1x LR")
    if int(cfg.SOLVER.WARMUP_ITERS) != 10:
        raise ValueError("all RGBNT201 paper rows must use 10 warm-up epochs")
    if str(cfg.SOLVER.SCHEDULER_UNIT).lower() != "epoch":
        raise ValueError("RGBNT201 paper warm-up and cosine schedule must use epochs")
    if float(cfg.INPUT.GRAY_REPLACE_PROB) != 0.0:
        raise ValueError("RGBNT201 paper rows must disable custom grayscale replacement")
    if float(cfg.INPUT.MODALITY_DROP_PROB) != 0.0:
        raise ValueError("RGBNT201 paper rows must disable modality dropout")
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


def run_row(args, timeout_bin, commit, experiment, row, row_config,
            output_name=None):
    output_dir = args.output_root / (output_name or OUTPUT_NAMES[row])
    output_dir.mkdir(parents=True)
    cfg = resolve_config(
        row_config, output_dir, seed=args.seed,
        expected_train_epochs=args.expected_train_epochs,
        expected_base_lr=args.expected_base_lr)
    output_dir.joinpath("resolved_config.yml").write_text(cfg.dump(), encoding="utf-8")
    output_dir.joinpath("commit.txt").write_text(commit + "\n", encoding="utf-8")
    command = [
        timeout_bin, "--signal=TERM", "--kill-after=10s", TIME_LIMIT,
        args.python, "train_net.py",
        "--config_file", BASE_CONFIG,
        "--config_file", row_config,
        "OUTPUT_DIR", str(output_dir),
        "SOLVER.EVAL_PERIOD", str(EVAL_PERIOD),
        "SOLVER.SEED", str(args.seed),
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
    parser.add_argument("--single-experiment")
    parser.add_argument("--single-row")
    parser.add_argument("--single-config")
    parser.add_argument("--single-output-name")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--expected-train-epochs", type=int, default=TRAIN_EPOCHS)
    parser.add_argument(
        "--expected-base-lr", type=float, default=3.5e-4)
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.expected_train_epochs <= 0:
        parser.error("--expected-train-epochs must be positive")
    if args.expected_base_lr <= 0:
        parser.error("--expected-base-lr must be positive")

    single_values = (
        args.single_experiment, args.single_row,
        args.single_config, args.single_output_name,
    )
    if any(single_values) and not all(single_values):
        parser.error("all --single-* arguments must be provided together")
    rows = ROWS
    output_names = dict(OUTPUT_NAMES)
    if all(single_values):
        rows = ((args.single_experiment, args.single_row, args.single_config),)
        output_names[args.single_row] = args.single_output_name

    plan = []
    for experiment, row, row_config in rows:
        output_dir = args.output_root / output_names[row]
        resolved_cfg = resolve_config(
            row_config, output_dir, seed=args.seed,
            expected_train_epochs=args.expected_train_epochs,
            expected_base_lr=args.expected_base_lr)
        plan.append({
            "experiment": experiment,
            "row": row,
            "base_config": BASE_CONFIG,
            "row_config": row_config,
            "output_dir": str(output_dir),
            "dataset": "RGBNT201",
            "seed": args.seed,
            "batch_size": BATCH_SIZE,
            "input_size": list(INPUT_SIZE),
            "epochs": int(resolved_cfg.SOLVER.TRAIN_EPOCHS),
            "scheduler_horizon_epochs": int(
                resolved_cfg.SOLVER.MAX_EPOCHS),
            "optimizer": resolved_cfg.SOLVER.OPTIMIZER_NAME,
            "base_lr": float(resolved_cfg.SOLVER.BASE_LR),
            "backbone_lr_factor": float(
                resolved_cfg.SOLVER.BACKBONE_LR_FACTOR),
            "backbone_lr": float(resolved_cfg.SOLVER.BASE_LR) * float(
                resolved_cfg.SOLVER.BACKBONE_LR_FACTOR),
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
    if all(single_values):
        experiment, row, row_config = rows[0]
        result = run_row(
            args, timeout_bin, commit, experiment, row, row_config,
            output_name=output_names[row])
        return 0 if result["status"] == "completed" else 1

    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_root / "fusion_E006-E008_{}".format(stamp)
    run_dir.mkdir(parents=True)
    run_dir.joinpath("plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    run_dir.joinpath("commit.txt").write_text(commit + "\n", encoding="utf-8")

    results = []
    for experiment, row, row_config in rows:
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
