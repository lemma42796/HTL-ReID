#!/usr/bin/env python3
"""CUDA smoke test for consensus-specific HS and its full-model wiring."""

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import cfg as default_cfg
from modeling import make_model
from modeling.fusion_part.HS import HS


def attention(scores, device):
    tensor = torch.zeros(1, 1, 4, 4, device=device)
    tensor[0, 0, 0, 1:] = torch.tensor(scores, device=device)
    return [tensor]


def assert_mask(actual, expected):
    expected_tensor = torch.tensor([expected], device=actual.device)
    if not torch.equal(actual, expected_tensor):
        raise AssertionError(
            'mask mismatch: {} != {}'.format(
                actual.int().tolist(), expected_tensor.int().tolist()))


def test_mask_semantics(device):
    features = [torch.randn(1, 4, 8, device=device) for _ in range(3)]
    attentions = [
        attention((0.9, 0.2, 0.1), device),
        attention((0.8, 0.3, 0.1), device),
        attention((0.1, 0.9, 0.2), device),
    ]
    selector = HS(ratio=1.0 / 3.0, consensus_specific=True).to(device)
    result = selector(
        features[0], attentions[0], features[1], attentions[1],
        features[2], attentions[2], return_gates=True)
    masks = result[3]
    assert_mask(masks[0], (True, False, False))
    assert_mask(masks[1], (True, False, False))
    assert_mask(masks[2], (True, True, False))

    legacy = HS(ratio=1.0 / 3.0).to(device)
    legacy_masks = legacy(
        features[0], attentions[0], features[1], attentions[1],
        features[2], attentions[2])[3]
    for mask in legacy_masks:
        assert_mask(mask, (True, True, False))


def build_cfg():
    cfg = default_cfg.clone()
    cfg.merge_from_file('configs/RGBNT201/paper/base.yml')
    cfg.merge_from_file(
        'configs/RGBNT201/fusion/a3_isolated_consensus_specific_clean.yml')
    cfg.MODEL.PRETRAIN_CHOICE = 'self'
    cfg.MODEL.PRETRAIN_PATH_T = ''
    cfg.INPUT.SIZE_TRAIN = [128, 64]
    cfg.INPUT.SIZE_TEST = [128, 64]
    cfg.SOLVER.IMS_PER_BATCH = 2
    return cfg


def test_full_model(device):
    torch.manual_seed(1111)
    cfg = build_cfg()
    model = make_model(cfg, num_class=4, camera_num=2).to(device)
    batch = {
        name: torch.randn(2, 3, 128, 64, device=device)
        for name in ('RGB', 'NI', 'TI')
    }
    camera = torch.zeros(2, dtype=torch.long, device=device)
    labels = torch.tensor([0, 1], dtype=torch.long, device=device)

    model.train()
    output = model(batch, cam_label=camera, label=labels, epoch=1)
    tensors = [value for value in output if torch.is_tensor(value)]
    if not tensors or not all(torch.isfinite(value).all() for value in tensors):
        raise AssertionError('full-model train output is missing or non-finite')
    sum(value.float().mean() for value in tensors).backward()
    if not any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()):
        raise AssertionError('full-model backward produced no finite gradient')

    model.eval()
    with torch.no_grad():
        descriptor = model(batch, cam_label=camera)
    if descriptor.shape != (2, 11520):
        raise AssertionError(
            'unexpected descriptor shape {}'.format(tuple(descriptor.shape)))
    if not torch.isfinite(descriptor).all():
        raise AssertionError('full-model descriptor is non-finite')


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; CPU fallback is forbidden')
    device = torch.device('cuda')
    test_mask_semantics(device)
    test_full_model(device)
    print('consensus-specific HS CUDA smoke: PASS')


if __name__ == '__main__':
    main()
