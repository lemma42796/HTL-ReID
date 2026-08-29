"""CUDA smoke test for the R001-R004 recovery configurations.

This test intentionally avoids datasets and pretrained checkpoints. It checks
config routing, model construction, three-modal forward/backward, finite
outputs, FACSS gradient flow, and the paper-cascade AGF gates.

Run on the remote GPU only:
    python tools/test_recovery_gpu.py
"""

import gc
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import cfg as default_cfg
from modeling.make_model import make_model


BASE = 'configs/RGBNT201/recovery/base.yml'
ROWS = {
    'R001': 'configs/RGBNT201/recovery/r001_backbone.yml',
    'R002': 'configs/RGBNT201/recovery/r002_hs.yml',
    'R003': 'configs/RGBNT201/recovery/r003_hs_facss.yml',
    'R003R': 'configs/RGBNT201/recovery/r003_facss_residual.yml',
    'R003V2': 'configs/RGBNT201/recovery/r003_facss_v2.yml',
    'R004S': 'configs/RGBNT201/recovery/r004_structural.yml',
    'R004': 'configs/RGBNT201/recovery/r004_full.yml',
}


def make_cfg(row):
    cfg = default_cfg.clone()
    cfg.merge_from_file(BASE)
    cfg.merge_from_file(ROWS[row])
    cfg.MODEL.PRETRAIN_CHOICE = 'self'
    cfg.MODEL.SIE_CAMERA = False
    cfg.INPUT.SIZE_TRAIN = [128, 64]
    cfg.INPUT.SIZE_TEST = [128, 64]
    return cfg


def assert_finite(value, name):
    if not torch.isfinite(value).all():
        raise AssertionError('{} contains NaN/Inf'.format(name))


def assert_any_finite_grad(parameters, name):
    grads = [p.grad for p in parameters if p.requires_grad and p.grad is not None]
    if not grads:
        raise AssertionError('{} received no gradients'.format(name))
    for grad in grads:
        assert_finite(grad, '{} gradient'.format(name))


def run_row(row):
    cfg = make_cfg(row)
    expected = {
        'R001': (False, False, False),
        'R002': (True, False, False),
        'R003': (True, True, False),
        'R003R': (True, True, True),
        'R003V2': (True, True, True),
        'R004S': (True, True, True),
        'R004': (True, True, True),
    }[row]
    actual = (
        bool(cfg.MODEL.HS_ENABLED),
        bool(cfg.MODEL.FACSS_ENABLED),
        bool(cfg.MODEL.AGF),
    )
    if actual != expected:
        raise AssertionError('{} switches {} != {}'.format(row, actual, expected))

    model = make_model(cfg, num_class=8, camera_num=0).cuda().train()
    batch = {
        name: torch.randn(2, 3, 128, 64, device='cuda')
        for name in ('RGB', 'NI', 'TI')
    }
    labels = torch.tensor([0, 1], device='cuda')
    output = model(batch, label=labels, epoch=1)
    expected_outputs = 11 if row == 'R004S' else 9
    if len(output) != expected_outputs:
        raise AssertionError('{} expected {} outputs, got {}'.format(
            row, expected_outputs, len(output)))
    if output[1].shape != (2, 3 * model.BACKBONE.token_dim):
        raise AssertionError('{} descriptor shape {}'.format(row, output[1].shape))
    for idx, value in enumerate(output):
        assert_finite(value, '{} output[{}]'.format(row, idx))

    loss = sum(value.float().mean() for value in output[:-1])
    loss.backward()
    assert_any_finite_grad(model.FUSE_HEAD.parameters(), '{} fused head'.format(row))
    if row in ('R003', 'R003R', 'R003V2', 'R004S', 'R004'):
        assert_any_finite_grad(
            model.HS_FACSS.alpha_mlp.parameters(), '{} FACSS alpha'.format(row))
    if row == 'R003V2':
        if model.HS_FACSS.cross_pool != 'dual_softmax':
            raise AssertionError('R003V2 did not construct dual-softmax FACSS')
        assert_any_finite_grad(
            model.HS_FACSS.match_q.parameters(), 'R003V2 FACSS query projection')
        assert_any_finite_grad(
            model.HS_FACSS.match_k.parameters(), 'R003V2 FACSS key projection')
    if row == 'R004S':
        if model.HS_FACSS.output_mode != 'soft_slots':
            raise AssertionError('R004S did not construct soft-slot FACSS')
        assert_any_finite_grad(
            [model.HS_FACSS.selector_queries], 'R004S FACSS slot queries')
        assert_any_finite_grad(
            model.LOCAL_HEAD.parameters(), 'R004S local identity head')
    if row in ('R003R', 'R003V2', 'R004S'):
        if model.AGF.mode != 'residual_cascade':
            raise AssertionError('{} did not construct residual_cascade fusion'.format(row))
        assert_any_finite_grad(
            model.AGF.residual_start.parameters(), '{} first residual stage'.format(row))
        assert_any_finite_grad(
            model.AGF.residual_middle.parameters(), '{} second residual stage'.format(row))
        assert_any_finite_grad(
            model.AGF.residual_end.parameters(), '{} self-aggregation stage'.format(row))
    if row == 'R004':
        if model.AGF.mode != 'paper_cascade':
            raise AssertionError('R004 did not construct paper_cascade AGF')
        assert_any_finite_grad(
            model.AGF.paper_start.parameters(), 'R004 first AGF stage')
        assert_any_finite_grad(
            model.AGF.paper_middle.parameters(), 'R004 second AGF stage')
        assert_any_finite_grad(
            model.AGF.paper_end.parameters(), 'R004 self-aggregation stage')

    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print('{} OK | switches={} | peak={:.1f} MiB'.format(row, actual, peak_mb))
    del output, loss, batch, model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; run this test on the remote GPU')
    torch.manual_seed(1111)
    torch.cuda.manual_seed_all(1111)
    for row in ROWS:
        run_row(row)
    print('RECOVERY GPU SMOKE PASSED')


if __name__ == '__main__':
    main()
