#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import os
import re
import subprocess
import sys
from pathlib import Path


ROWS = [
    ("A0", "backbone", "configs/RGBNT201/ablations/chain20/a0_backbone.yml"),
    ("A1", "hs_facss", "configs/RGBNT201/ablations/chain20/a1_hs_facss.yml"),
    ("A2", "quality", "configs/RGBNT201/ablations/chain20/a2_quality.yml"),
    ("A3", "agf", "configs/RGBNT201/ablations/chain20/a3_agf.yml"),
    ("A4", "adapter", "configs/RGBNT201/ablations/chain20/a4_adapter.yml"),
    ("A5", "full", "configs/RGBNT201/ablations/chain20/a5_full.yml"),
]

METRIC_PATTERNS = {
    "map": re.compile(r"mAP:\s+([0-9.]+)%"),
    "rank1": re.compile(r"Rank-1\s*:?\s*([0-9.]+)%"),
    "rank5": re.compile(r"Rank-5\s*:?\s*([0-9.]+)%"),
    "rank10": re.compile(r"Rank-10\s*:?\s*([0-9.]+)%"),
}
EPOCH_PATTERN = re.compile(r"Validation Results - Epoch:\s+(\d+)")


def parse_best_metrics(log_path):
    best = {
        "best_epoch": "",
        "best_mAP": "",
        "best_Rank1": "",
        "best_Rank5": "",
        "best_Rank10": "",
    }
    current = {"epoch": ""}
    validations = []
    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            epoch_match = EPOCH_PATTERN.search(line)
            if epoch_match:
                current = {"epoch": epoch_match.group(1)}
                validations.append(current)
                continue
            if not validations:
                continue
            for key, pattern in METRIC_PATTERNS.items():
                metric_match = pattern.search(line)
                if metric_match:
                    current[key] = metric_match.group(1)

    complete = [item for item in validations if "map" in item]
    if complete:
        top = max(complete, key=lambda item: float(item["map"]))
        best = {
            "best_epoch": top.get("epoch", ""),
            "best_mAP": top.get("map", ""),
            "best_Rank1": top.get("rank1", ""),
            "best_Rank5": top.get("rank5", ""),
            "best_Rank10": top.get("rank10", ""),
        }
    return best


def write_summary(summary_path, rows):
    fieldnames = [
        "row",
        "name",
        "config",
        "output_dir",
        "returncode",
        "best_epoch",
        "best_mAP",
        "best_Rank1",
        "best_Rank5",
        "best_Rank10",
    ]
    with open(summary_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_row(args, root, row_id, name, config_path):
    output_dir = root / f"{row_id}_{name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "stdout.log"
    cmd = [
        args.python,
        "train_net.py",
        "--config_file",
        "configs/RGBNT201/default.yml",
        "--config_file",
        config_path,
        "OUTPUT_DIR",
        str(output_dir),
    ]
    (output_dir / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(" ".join(cmd) + "\n\n")
        log_file.flush()
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        returncode = proc.wait()
    metrics = parse_best_metrics(log_path)
    return {
        "row": row_id,
        "name": name,
        "config": config_path,
        "output_dir": str(output_dir),
        "returncode": returncode,
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="/root/miniconda3/bin/python")
    parser.add_argument("--output-root", default="")
    args = parser.parse_args()

    if args.output_root:
        root = Path(args.output_root)
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        root = Path("/root/autodl-tmp/outputs/HTL-ReID") / f"chain20-RGBNT201-{stamp}"
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "summary.csv"
    rows = []

    for row_id, name, config_path in ROWS:
        result = run_row(args, root, row_id, name, config_path)
        rows.append(result)
        write_summary(summary_path, rows)
        if result["returncode"] != 0:
            (root / "FAILED").write_text(
                f"{row_id}_{name} failed with return code {result['returncode']}\n",
                encoding="utf-8",
            )
            return result["returncode"]

    (root / "DONE").write_text("all rows finished\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
