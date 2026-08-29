#!/usr/bin/env python3
"""CUDA-only forward/backward smoke check for E057 and E058."""

from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import cfg as default_cfg
from modeling.make_model import make_model


BASE = 'configs/RGBNT201/paper/base.yml'
ROWS = (
    ('legacy', 'configs/RGBNT201/fusion/t14_clean_baseline.yml', 16896),
    ('cruf', 'configs/RGBNT201/fusion/t15_cruf_clean.yml', 3072),
)


def build_cfg(row):
    cfg = default_cfg.clone()
    cfg.merge_from_file(BASE)
    cfg.merge_from_file(row)
    cfg.MODEL.PRETRAIN_CHOICE = 'self'
    cfg.MODEL.SIE_CAMERA = False
    cfg.INPUT.SIZE_TRAIN = [128, 64]
    cfg.INPUT.SIZE_TEST = [128, 64]
    return cfg


def dummy_batch(device):
    return {
        name: torch.randn(4, 3, 128, 64, device=device)
        for name in ('RGB', 'NI', 'TI')
    }


def smoke(name, row, expected_dim):
    cfg = build_cfg(row)
    model = make_model(cfg, num_class=2, camera_num=0).cuda().train()
    inputs = dummy_batch(torch.device('cuda'))
    labels = torch.tensor([0, 0, 1, 1], device='cuda')
    output = model(inputs, label=labels, epoch=1)
    if len(output) != 9:
        raise AssertionError(
            '{} returned {} training values, expected 9'.format(
                name, len(output)))
    if not all(torch.isfinite(value).all() for value in output):
        raise AssertionError('{} produced NaN/Inf'.format(name))
    loss = sum(value.float().square().mean() for value in output[:-1])
    loss = loss + output[-1].float()
    loss.backward()

    if name == 'cruf':
        required = (
            'DECOUPLED_MOE.gate_query.weight',
            'DECOUPLED_MOE.route_tokens',
            'CRUF_FINAL_HEAD.weight',
        )
        parameters = dict(model.named_parameters())
        for parameter_name in required:
            gradient = parameters[parameter_name].grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise AssertionError(
                    '{} lacks a finite gradient'.format(parameter_name))
        if model.DECOUPLED_MOE.output_dim != model.BACKBONE.token_dim:
            raise AssertionError('CRUF did not reduce the route output to 768D')

    model.eval()
    with torch.no_grad():
        descriptor = model(inputs, epoch=1)
    if descriptor.shape != (4, expected_dim):
        raise AssertionError(
            '{} descriptor shape {}, expected {}'.format(
                name, tuple(descriptor.shape), expected_dim))
    torch.cuda.synchronize()
    print('OK {} descriptor={} aux={:.6f} memory={:.1f} MiB'.format(
        name, tuple(descriptor.shape), float(output[-1]),
        torch.cuda.max_memory_allocated() / 2 ** 20))


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; CPU fallback is disabled')
    print(torch.cuda.get_device_name(0))
    for name, row, expected_dim in ROWS:
        torch.cuda.reset_peak_memory_stats()
        smoke(name, row, expected_dim)


if __name__ == '__main__':
    main()
