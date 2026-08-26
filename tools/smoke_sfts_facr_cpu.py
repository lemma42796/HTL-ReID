"""CPU regression checks for SFTS residual tokens and FACR self-refinement."""

import argparse
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import cfg as default_cfg
from modeling.fusion_part.SFTS import SFTS
from modeling.fusion_part.TPM import FACR
from modeling.make_model import make_model


BASE = 'configs/RGBNT201/paper/base.yml'
ROWS = (
    'configs/RGBNT201/fusion/t9_facr_self_refine.yml',
    'configs/RGBNT201/fusion/t10_sfts_k1_residual_facr.yml',
)


def attention_stack(batch, heads, tokens, layers=3):
    return [
        torch.softmax(torch.randn(batch, heads, tokens, tokens), dim=-1)
        for _ in range(layers)
    ]


def check_configs():
    for row in ROWS:
        cfg = default_cfg.clone()
        cfg.merge_from_file(BASE)
        cfg.merge_from_file(row)
        assert cfg.MODEL.FACR
        assert cfg.MODEL.FACR_SELF_REFINE
    print('OK config merge')


def check_sfts_residual_gradient():
    torch.manual_seed(11)
    batch, heads, patches, dim = 2, 2, 16, 16
    features = [
        torch.randn(batch, patches + 1, dim, requires_grad=True)
        for _ in range(3)
    ]
    attentions = [
        attention_stack(batch, heads, patches + 1)
        for _ in range(3)
    ]
    selector = SFTS(ratio=1.0 / patches)
    legacy = selector(
        features[0], attentions[0],
        features[1], attentions[1],
        features[2], attentions[2],
        return_gates=True)
    assert len(legacy) == 5

    output = selector(
        features[0], attentions[0],
        features[1], attentions[1],
        features[2], attentions[2],
        return_gates=True, return_residual_tokens=True)
    assert len(output) == 6
    masks = output[3]
    residual_tokens = output[-1]
    assert len(residual_tokens) == 3
    assert all(token.shape == (batch, dim) for token in residual_tokens)
    assert all(torch.isfinite(token).all() for token in residual_tokens)

    sum(token.sum() for token in residual_tokens).backward()
    for feature, mask in zip(features, masks):
        dropped_grad = feature.grad[:, 1:, :][~mask]
        assert dropped_grad.numel() > 0
        assert dropped_grad.abs().sum() > 0
    print('OK SFTS residual-token gradient')


def check_facr_self_refinement():
    torch.manual_seed(17)
    batch, patches, dim = 2, 8, 16
    features = tuple(
        torch.randn(batch, patches + 1, dim, requires_grad=True)
        for _ in range(3)
    )
    masks = tuple(
        torch.tensor(
            [[1, 1, 0, 0, 1, 0, 0, 0],
             [1, 0, 1, 0, 0, 1, 0, 0]],
            dtype=torch.bool)
        for _ in range(3)
    )
    residual_tokens = tuple(
        torch.randn(batch, dim, requires_grad=True)
        for _ in range(3)
    )
    model = FACR(
        dim=dim, num_heads=4, steps=2,
        score_bias_scale=0.0, self_refine=True,
        self_refine_scale_init=0.1)
    descriptor = model(
        *features, masks=masks, residual_tokens=residual_tokens)
    assert descriptor.shape == (batch, 3 * dim)
    assert torch.isfinite(descriptor).all()
    descriptor.square().mean().backward()
    parameters = [p for p in model.parameters() if p.requires_grad]
    assert parameters
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in parameters)
    assert all(token.grad is not None and token.grad.abs().sum() > 0
               for token in residual_tokens)
    print('OK FACR self-refinement forward/backward')


def check_disabled_contract():
    model = FACR(dim=16, num_heads=4, steps=1, self_refine=False)
    assert not hasattr(model, 'self_refinement')
    features = tuple(torch.randn(2, 9, 16) for _ in range(3))
    descriptor = model(*features)
    assert descriptor.shape == (2, 48)
    print('OK disabled-path contract')


def check_full_model():
    row = 'configs/RGBNT201/fusion/t10_sfts_k1_residual_facr.yml'
    cfg = default_cfg.clone()
    cfg.merge_from_file(BASE)
    cfg.merge_from_file(row)
    cfg.MODEL.PRETRAIN_CHOICE = 'self'
    cfg.MODEL.SIE_CAMERA = False
    cfg.INPUT.SIZE_TRAIN = [64, 64]
    cfg.INPUT.SIZE_TEST = [64, 64]
    model = make_model(cfg, num_class=8, camera_num=0).train()
    for parameter in model.BACKBONE.parameters():
        parameter.requires_grad_(False)
    inputs = {
        name: torch.randn(2, 3, 64, 64)
        for name in ('RGB', 'NI', 'TI')
    }
    labels = torch.randint(0, 8, (2,))
    output = model(inputs, label=labels, epoch=1)
    assert len(output) == 5
    assert all(torch.isfinite(value).all() for value in output)
    loss = sum(value.float().mean() for value in output[:-1]) + output[-1]
    loss.backward()
    self_refine = [
        parameter for name, parameter in model.named_parameters()
        if 'FACR.self_refinement' in name and parameter.requires_grad
    ]
    assert self_refine
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in self_refine)
    model.eval()
    with torch.no_grad():
        descriptor = model(inputs, epoch=1)
    assert descriptor.shape == (2, 3 * model.BACKBONE.token_dim)
    assert torch.isfinite(descriptor).all()
    print('OK full-model T10 train/eval integration')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full-model', action='store_true')
    args = parser.parse_args()
    check_configs()
    check_sfts_residual_gradient()
    check_facr_self_refinement()
    check_disabled_contract()
    if args.full_model:
        check_full_model()


if __name__ == '__main__':
    main()
