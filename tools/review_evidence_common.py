"""Shared helpers for final, inference-only reviewer evidence."""

import random
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.defaults import _C as DEFAULT_CFG
from data import make_dataloader
from modeling import make_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_cfg(base_config, row_config, seed=1111, test_batch_size=64):
    cfg = DEFAULT_CFG.clone()
    cfg.merge_from_file(str(base_config))
    cfg.merge_from_file(str(row_config))
    cfg.SOLVER.SEED = int(seed)
    cfg.TEST.RE_RANKING = 'no'
    cfg.TEST.IMS_PER_BATCH = int(test_batch_size)
    return cfg


def build_eval_run(base_config, row_config, checkpoint, seed=1111,
                   test_batch_size=64):
    cfg = build_cfg(
        base_config, row_config, seed=seed,
        test_batch_size=test_batch_size)
    loaders = make_dataloader(cfg)
    train_loader, train_loader_normal, val_loader = loaders[:3]
    num_query, num_classes, camera_num = loaders[3:6]
    del train_loader, train_loader_normal
    cfg.MODEL.PRETRAIN_CHOICE = 'self'
    cfg.MODEL.PRETRAIN_PATH_T = ''
    model = make_model(
        cfg, num_class=num_classes, camera_num=camera_num).cuda()
    model.load_param(str(checkpoint))
    model.eval()
    return cfg, model, val_loader, num_query


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; CPU fallback is forbidden')
