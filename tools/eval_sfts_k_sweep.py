#!/usr/bin/env python3
"""Evaluate one learnable-K HS checkpoint at several forced K values."""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import cfg
from data import make_dataloader
from engine.processor import _make_evaluator
from modeling import make_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(cfg, model, val_loader, num_query):
    evaluator = _make_evaluator(cfg, num_query)
    evaluator.reset()
    started = time.monotonic()
    model.eval()
    with torch.inference_mode():
        for images, vid, camid, camids, target_view, paths in val_loader:
            images = {
                "RGB": images["RGB"].cuda(non_blocking=True),
                "NI": images["NI"].cuda(non_blocking=True),
                "TI": images["TI"].cuda(non_blocking=True),
            }
            feature = model(
                images,
                cam_label=camids.cuda(non_blocking=True),
                view_label=target_view.cuda(non_blocking=True),
                mode=1,
                img_path=paths,
            )
            evaluator.update((feature, vid, camid))
    cmc, mean_ap, *_ = evaluator.compute(cfg)
    return {
        "mAP": 100.0 * float(mean_ap),
        "Rank1": 100.0 * float(cmc[0]),
        "Rank5": 100.0 * float(cmc[4]),
        "Rank10": 100.0 * float(cmc[9]),
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    args = parser.parse_args()

    for config_file in args.config_file:
        cfg.merge_from_file(config_file)
    if str(cfg.TEST.RE_RANKING).lower() != "no":
        raise ValueError("K sweep must run with re-ranking disabled")
    if not cfg.MODEL.HS_ENABLED or not cfg.MODEL.HS_LEARNABLE_K:
        raise ValueError("K sweep requires a learnable-K HS config")

    candidates = [int(value) for value in cfg.MODEL.HS_K_CANDIDATES]
    requested = [int(value) for value in args.k]
    missing = sorted(set(requested) - set(candidates))
    if missing:
        raise ValueError("requested K values absent from checkpoint config: {}".format(missing))
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    set_seed(int(cfg.SOLVER.SEED))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    del train_loader, train_loader_normal, view_num
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num)
    model.load_param(str(args.checkpoint))
    model.cuda()

    results = []
    with torch.no_grad():
        for keep_k in requested:
            model.HS.k_logits.fill_(-100.0)
            model.HS.k_logits[candidates.index(keep_k)] = 100.0
            metrics = evaluate(cfg, model, val_loader, num_query)
            result = {"K": keep_k, **metrics}
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)

    payload = {
        "checkpoint": str(args.checkpoint),
        "configs": args.config_file,
        "candidates": candidates,
        "re_ranking": str(cfg.TEST.RE_RANKING),
        "results": results,
        "best_by_mAP": max(results, key=lambda item: item["mAP"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("saved {}".format(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
