#!/usr/bin/env python3
"""Measure the actual post-union SFTS token retention of one checkpoint."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import cfg as default_cfg
from data import make_dataloader
from modeling import make_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cfg = default_cfg.clone()
    for config_file in args.config_file:
        cfg.merge_from_file(config_file)
    if not cfg.MODEL.SFTS_ENABLED:
        raise ValueError("retention measurement requires SFTS")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    set_seed(int(cfg.SOLVER.SEED))
    train_loader, train_loader_normal, val_loader, _, num_classes, camera_num, _ = make_dataloader(cfg)
    del train_loader, train_loader_normal
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num)
    model.load_param(str(args.checkpoint))
    model.cuda().eval()

    counts = []

    def capture_mask(_module, _inputs, output):
        masks = output[3]
        mask = masks[0] if isinstance(masks, (tuple, list)) else masks
        counts.extend(mask.detach().sum(dim=1).cpu().tolist())

    handle = model.SFTS.register_forward_hook(capture_mask)
    with torch.inference_mode():
        for images, _, _, camids, target_view, paths in val_loader:
            images = {
                name: tensor.cuda(non_blocking=True)
                for name, tensor in images.items()
            }
            model(
                images,
                cam_label=camids.cuda(non_blocking=True),
                view_label=target_view.cuda(non_blocking=True),
                mode=1,
                img_path=paths,
            )
    handle.remove()

    values = torch.tensor(counts, dtype=torch.float64)
    patch_count = int(model.num_patches)
    result = {
        "checkpoint": str(args.checkpoint),
        "configs": args.config_file,
        "samples": int(values.numel()),
        "patches_per_sample": patch_count,
        "mean_retained_tokens": float(values.mean()),
        "retention_percent": 100.0 * float(values.mean()) / patch_count,
        "min_retained_tokens": int(values.min()),
        "max_retained_tokens": int(values.max()),
    }
    payload = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
