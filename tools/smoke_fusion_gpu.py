"""Short CUDA-only forward/backward smoke test for TPM/FACR configs."""

import argparse
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
    'configs/RGBNT201/fusion/t1_tpm.yml',
    'configs/RGBNT201/fusion/t2_adaptive_routing.yml',
    'configs/RGBNT201/fusion/t3_m2_facr.yml',
    'configs/RGBNT201/fusion/t4_facss_masked_facr.yml',
    'configs/RGBNT201/fusion/t5_sfts_masked_facr.yml',
    'configs/RGBNT201/fusion/t6_sfts_learnable_k_facr.yml',
    'configs/RGBNT201/fusion/t7_sfts_fixed_k16_facr.yml',
    'configs/RGBNT201/fusion/t8_sfts_fixed_k16_route_balance.yml',
    'configs/RGBNT201/fusion/t9_facr_self_refine.yml',
    'configs/RGBNT201/fusion/t10_sfts_k1_residual_facr.yml',
)


def build_cfg(row, height, width):
    cfg = default_cfg.clone()
    cfg.merge_from_file(BASE)
    cfg.merge_from_file(row)
    cfg.MODEL.PRETRAIN_CHOICE = 'self'
    cfg.MODEL.SIE_CAMERA = False
    cfg.INPUT.SIZE_TRAIN = [height, width]
    cfg.INPUT.SIZE_TEST = [height, width]
    return cfg


def dummy_batch(batch, height, width, device):
    return {
        name: torch.randn(batch, 3, height, width, device=device)
        for name in ('RGB', 'NI', 'TI')
    }


def smoke(row, batch, height, width):
    cfg = build_cfg(row, height, width)
    model = make_model(cfg, num_class=8, camera_num=0).cuda().train()
    inputs = dummy_batch(batch, height, width, torch.device('cuda'))
    labels = torch.randint(0, 8, (batch,), device='cuda')
    output = model(inputs, label=labels, epoch=0)
    if len(output) != 5:
        raise AssertionError('{} returned {} values, expected 5'.format(row, len(output)))
    loss = sum(value.float().mean() for value in output[:-1]) + output[-1].float()
    loss.backward()
    if not all(torch.isfinite(value).all() for value in output):
        raise AssertionError('{} produced NaN/Inf'.format(row))
    fusion_parameters = [
        parameter for name, parameter in model.named_parameters()
        if ('TPM' in name or 'FACR' in name) and parameter.requires_grad
    ]
    if not fusion_parameters or not all(parameter.grad is not None for parameter in fusion_parameters):
        raise AssertionError('{} fusion parameters did not all receive gradients'.format(row))
    if cfg.MODEL.FACR and cfg.MODEL.FACR_USE_SCORES:
        score_parameters = [
            parameter for name, parameter in model.named_parameters()
            if 'HS_FACSS.alpha_mlp' in name and parameter.requires_grad
        ]
        if not score_parameters or not all(parameter.grad is not None for parameter in score_parameters):
            raise AssertionError('{} FACSS score networks did not receive gradients'.format(row))
    if cfg.MODEL.SFTS_LEARNABLE_K:
        k_logits = model.SFTS.k_logits
        if (k_logits.grad is None or not torch.isfinite(k_logits.grad).all() or
                k_logits.grad.abs().sum() == 0):
            raise AssertionError(
                '{} learnable K did not receive finite non-zero gradients'.format(row))
    model.eval()
    with torch.no_grad():
        descriptor = model(inputs, epoch=0)
    expected = 3 * model.BACKBONE.token_dim
    if descriptor.shape != (batch, expected):
        raise AssertionError('{} descriptor shape {}'.format(row, descriptor.shape))
    torch.cuda.synchronize()
    print('OK {} descriptor={} memory={:.1f} MiB'.format(
        row, tuple(descriptor.shape), torch.cuda.max_memory_allocated() / 2**20))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--height', type=int, default=128)
    parser.add_argument('--width', type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; CPU fallback is intentionally disabled')
    print(torch.cuda.get_device_name(0))
    for row in ROWS:
        torch.cuda.reset_peak_memory_stats()
        smoke(row, args.batch, args.height, args.width)


if __name__ == '__main__':
    main()
