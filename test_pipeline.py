"""
End-to-end pipeline smoke test for HTL-ReID.

Runs on CPU. Verifies that a fresh checkout with the published configs can
build the model and complete forward + backward without CUDA, real data, or
pretrained weights. This is the test others should run after `pip install
-r requirements.txt` to confirm their setup.

Coverage:
  1. cfg defaults load and freeze
  2. Each shipped yml config merges cleanly into defaults
  3. Iteration-based scheduler warmup semantics
  4. Model builds for each yml (PRETRAIN_CHOICE forced off so no .pth needed)
  5. 3-modal training forward returns the right tuple length and shapes
  6. 3-modal eval forward
  7. 2-modal forward_two_modalities (RGBN300-style path)
  8. Optional test-time part descriptor
  9. Backward populates non-NaN gradients on every trainable parameter
 10. Loss assembly matches engine/processor.py's odd/even pairing rule
 11. state_dict save -> reload -> identical forward output
 12. Ablation switches (AGF=0, OCFR=1) each produce a usable model
 13. Paper M0-M3 configs differ only through explicit HS/FACSS/QAWF switches
 14. M0 bypass, HS-only, and HS+FACSS selection paths behave distinctly
 15. Every paper row completes an end-to-end train/eval smoke pass

Run:
    python3 test_pipeline.py
"""
import copy
import io
import sys
import torch

from config import cfg as default_cfg
from modeling.make_model import make_model
from modeling.fusion_part.HS_FACSS import HSFACSS
from solver.scheduler_factory import create_scheduler


YMLS = [
    'configs/RGBNT201/default.yml',
    'configs/Market1501-MM/default.yml',
    'configs/MSVR310/default.yml',
    'configs/RGBNT100/default.yml',
]
PAPER_BASE = 'configs/RGBNT201/paper/base.yml'
PAPER_ROWS = {
    'M0': 'configs/RGBNT201/paper/m0.yml',
    'M1': 'configs/RGBNT201/paper/m1.yml',
    'M2': 'configs/RGBNT201/paper/m2.yml',
    'M3': 'configs/RGBNT201/paper/m3.yml',
}

NUM_CLASSES = 8
BATCH = 2


def _make_cfg(yml=None, **overrides):
    c = default_cfg.clone()
    if yml is not None:
        c.merge_from_file(yml)
    c.MODEL.PRETRAIN_CHOICE = 'self'        # skip ImageNet weight load
    c.MODEL.SIE_CAMERA = False              # avoid camera_num plumbing in tests
    c.INPUT.SIZE_TRAIN = [128, 64]           # keep CPU smoke tests quick
    c.INPUT.SIZE_TEST = [128, 64]
    for k, v in overrides.items():
        # supports dotted MODEL.XYZ overrides
        node = c
        parts = k.split('.')
        for p in parts[:-1]:
            node = getattr(node, p)
        setattr(node, parts[-1], v)
    return c


def _make_paper_cfg(row):
    c = default_cfg.clone()
    c.merge_from_file(PAPER_BASE)
    c.merge_from_file(PAPER_ROWS[row])
    return c


def _dummy_batch(cfg, modalities=('RGB', 'NI', 'TI')):
    H, W = cfg.INPUT.SIZE_TRAIN
    return {m: torch.randn(BATCH, 3, H, W) for m in modalities}


def _assert_finite(t, name):
    assert torch.isfinite(t).all(), '{} has NaN/Inf'.format(name)


def _expected_eval_dim(cfg, model):
    cls_dim = 3 * model.BACKBONE.token_dim
    part_dim = 3 * cfg.MODEL.PART_NUM * model.BACKBONE.token_dim
    if not cfg.MODEL.PART_BRANCH or cfg.TEST.PART_FEAT == 'off':
        return cls_dim
    if cfg.TEST.PART_FEAT == 'only':
        return part_dim
    return cls_dim + part_dim


def _loss_assembly_like_processor(output):
    """Lightweight trainability surrogate using the trainer's pair layout."""
    loss = torch.zeros(())
    if len(output) % 2 == 1:
        for i in range(0, len(output) - 1, 2):
            loss = loss + output[i].sum() + output[i + 1].sum()
        loss = loss + output[-1]
    else:
        for i in range(0, len(output), 2):
            loss = loss + output[i].sum() + output[i + 1].sum()
    return loss


def test_defaults_load():
    print('[1] cfg defaults clone+freeze')
    c = default_cfg.clone()
    c.freeze()
    assert c.MODEL.AGF in (0, 1)
    assert c.MODEL.HS_ENABLED in (0, 1)
    assert c.MODEL.FACSS_ENABLED in (0, 1)
    assert c.MODEL.FREQUENCY_ENABLED in (0, 1)
    print('     OK')


def test_yml_configs_merge():
    print('[2] each yml merges into defaults')
    for y in YMLS:
        c = default_cfg.clone()
        c.merge_from_file(y)
        c.freeze()
        print('     OK: {}  AGF={} AL={}'.format(
            y, c.MODEL.AGF, c.MODEL.AL))


def test_iteration_scheduler():
    print('[3] iteration scheduler semantics')
    c = default_cfg.clone()
    c.SOLVER.MAX_EPOCHS = 2
    c.SOLVER.WARMUP_ITERS = 3
    c.SOLVER.SCHEDULER_UNIT = 'iteration'
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=c.SOLVER.BASE_LR)
    scheduler = create_scheduler(c, opt, num_batches=5)
    assert scheduler.t_in_epochs is False
    scheduler.step_update(0)
    lr0 = opt.param_groups[0]['lr']
    scheduler.step_update(3)
    lr3 = opt.param_groups[0]['lr']
    assert lr3 > lr0, 'warmup lr did not increase: {} -> {}'.format(lr0, lr3)
    print('     OK lr {:.2e} -> {:.2e}'.format(lr0, lr3))


def test_three_modal_pipeline(yml):
    print('[4] 3-modal train+eval | {}'.format(yml))
    cfg = _make_cfg(yml)
    model = make_model(cfg, num_class=NUM_CLASSES, camera_num=0).cpu()
    model.train()

    x = _dummy_batch(cfg)
    label = torch.randint(0, NUM_CLASSES, (BATCH,))

    output = model(x, cam_label=None, label=label, epoch=0)
    assert isinstance(output, tuple), 'expected tuple, got {}'.format(type(output))

    # AL=1: (score, cls4t, ori_score, ori, part_score, part_feat, loss_aux) = 7
    # AL=0: (score, cls4t, RGB_s, RGB_f, NIR_s, NIR_f, TIR_s, TIR_f,
    #        part_score, part_feat, loss_aux) = 11
    expected_len = 7 if cfg.MODEL.AL else 11
    assert len(output) == expected_len, 'expected {} outputs, got {}'.format(
        expected_len, len(output))

    score, cls4t = output[0], output[1]
    assert score.shape == (BATCH, NUM_CLASSES), 'score shape {}'.format(score.shape)
    assert cls4t.shape == (BATCH, 3 * model.BACKBONE.token_dim), \
        'cls4t shape {}'.format(cls4t.shape)
    for i, t in enumerate(output):
        _assert_finite(t, 'output[{}]'.format(i))

    # Backward via the same loss assembly the trainer uses
    loss = _loss_assembly_like_processor(output)
    loss.backward()

    bad = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            bad.append((n, 'grad is None'))
        elif not torch.isfinite(p.grad).all():
            bad.append((n, 'NaN/Inf in grad'))
    # Some params are intentionally unused in this forward path. Filter them.
    # - AL_HEAD/BN: only used when MODEL.AL=1
    # - BACKBONE_HEAD/BN: only used when MODEL.AL=0
    # - BACKBONE.base.fc: ViT's ImageNet classifier, never used (we use embeddings)
    unused_ok = ('AL_HEAD', 'AL_BN', 'BACKBONE_HEAD', 'BACKBONE_BN', 'BACKBONE.base.fc')
    bad = [b for b in bad
           if not (b[1] == 'grad is None' and any(u in b[0] for u in unused_ok))]
    assert not bad, 'gradient issues:\n  ' + '\n  '.join('{} -- {}'.format(*b) for b in bad)
    print('     train fwd+bwd OK ({} outputs, all grads finite)'.format(len(output)))

    # Eval
    model.eval()
    with torch.no_grad():
        cls_eval = model(x, cam_label=None, epoch=0)
    assert cls_eval.shape == (BATCH, _expected_eval_dim(cfg, model))
    _assert_finite(cls_eval, 'eval cls4t')
    print('     eval fwd OK shape={}'.format(tuple(cls_eval.shape)))


def test_two_modal_pipeline():
    print('[5] 2-modal forward_two_modalities (AL=0)')
    cfg = _make_cfg('configs/RGBNT201/default.yml', **{'MODEL.AL': 0})
    model = make_model(cfg, num_class=NUM_CLASSES, camera_num=0).cpu()
    model.train()

    x = _dummy_batch(cfg, modalities=('RGB', 'NI'))
    label = torch.randint(0, NUM_CLASSES, (BATCH,))
    output = model.forward_two_modalities(x, cam_label=None, label=label, epoch=0)

    # AL=0 2-modal: (score, cls4t, RGB_s, RGB_f, NIR_s, NIR_f,
    #                 part_score, part_feat, loss_aux) = 9
    assert len(output) == 9, 'expected 9 outputs, got {}'.format(len(output))
    score, cls4t = output[0], output[1]
    assert score.shape == (BATCH, NUM_CLASSES)
    assert cls4t.shape == (BATCH, 3 * model.BACKBONE.token_dim)
    for i, t in enumerate(output):
        _assert_finite(t, 'output[{}]'.format(i))
    loss = _loss_assembly_like_processor(output)
    loss.backward()
    print('     train fwd+bwd OK ({} outputs)'.format(len(output)))

    model.eval()
    with torch.no_grad():
        cls_eval = model.forward_two_modalities(x, cam_label=None, epoch=0)
    assert cls_eval.shape == (BATCH, _expected_eval_dim(cfg, model))
    print('     eval fwd OK')


def test_part_descriptor_mode():
    print('[6] optional test-time part descriptor')
    cfg = _make_cfg('configs/RGBNT201/default.yml', **{'TEST.PART_FEAT': 'concat'})
    model = make_model(cfg, num_class=NUM_CLASSES, camera_num=0).cpu()
    model.eval()
    x = _dummy_batch(cfg)
    with torch.no_grad():
        feat = model(x, cam_label=None, epoch=0)
    expected_dim = 3 * model.BACKBONE.token_dim + 3 * cfg.MODEL.PART_NUM * model.BACKBONE.token_dim
    assert feat.shape == (BATCH, expected_dim), 'part concat shape {}'.format(feat.shape)
    _assert_finite(feat, 'part concat feat')
    print('     eval fwd OK shape={}'.format(tuple(feat.shape)))


def test_save_load_roundtrip():
    print('[7] state_dict save/load round-trip preserves output')
    cfg = _make_cfg('configs/RGBNT201/default.yml')
    torch.manual_seed(42)
    model = make_model(cfg, num_class=NUM_CLASSES, camera_num=0).cpu()
    model.eval()
    x = _dummy_batch(cfg)
    with torch.no_grad():
        out_before = model(x, cam_label=None, epoch=0)

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)
    sd = torch.load(buf, map_location='cpu', weights_only=True)

    torch.manual_seed(42)
    model2 = make_model(cfg, num_class=NUM_CLASSES, camera_num=0).cpu()
    # Use the model's own load_param so we exercise the public API
    buf2 = io.BytesIO()
    torch.save(sd, buf2)
    buf2.seek(0)
    # write to a temp file because load_param expects a path-like
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as fh:
        torch.save(sd, fh.name)
        path = fh.name
    model2.load_param(path)
    model2.eval()
    with torch.no_grad():
        out_after = model2(x, cam_label=None, epoch=0)
    diff = (out_before - out_after).abs().max().item()
    assert diff < 1e-5, 'roundtrip diff {} > 1e-5'.format(diff)
    print('     OK (max abs diff = {:.2e})'.format(diff))


def test_ablation_switches():
    print('[8] ablation switches')
    base_yml = 'configs/RGBNT201/default.yml'
    matrix = [
        {'MODEL.AGF': 0, 'MODEL.OCFR': 0},
        {'MODEL.AGF': 1, 'MODEL.OCFR': 1},
    ]
    for m in matrix:
        cfg = _make_cfg(base_yml, **m)
        model = make_model(cfg, num_class=NUM_CLASSES, camera_num=0).cpu()
        model.train()
        x = _dummy_batch(cfg)
        label = torch.randint(0, NUM_CLASSES, (BATCH,))
        out = model(x, cam_label=None, label=label, epoch=0)
        loss = _loss_assembly_like_processor(out)
        loss.backward()
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print('     OK  AGF={} OCFR={}  params={:.2f}M'.format(
            m['MODEL.AGF'], m['MODEL.OCFR'], n_params))


def test_paper_config_matrix():
    print('[9] formal paper config matrix')
    expected = {
        'M0': (0, 0, 0, 0.0),
        'M1': (1, 0, 0, 0.15),
        'M2': (1, 1, 0, 0.15),
        'M3': (1, 1, 1, 0.15),
    }
    for row, values in expected.items():
        c = _make_paper_cfg(row)
        actual = (
            int(c.MODEL.HS_ENABLED),
            int(c.MODEL.FACSS_ENABLED),
            int(c.MODEL.QUALITY_AWARE),
            float(c.MODEL.SELECTED_PATCH_BLEND_WEIGHT),
        )
        assert actual == values, '{} switches {} != {}'.format(row, actual, values)
        assert not c.MODEL.FREQUENCY_ENABLED
        assert not c.MODEL.AGF
        assert not c.MODEL.MODALITY_ADAPTER
        assert not c.MODEL.PART_BRANCH
        assert not c.MODEL.OCFR
        assert c.MODEL.ALIGN_LOSS_WEIGHT == 0
        assert c.MODEL.TOKEN_CONSISTENCY_WEIGHT == 0
        assert c.MODEL.BCC_LOSS_WEIGHT == 0
        assert c.MODEL.AUX_LOSS_WEIGHT == 0
        assert c.TEST.RE_RANKING == 'no'
        assert c.SOLVER.SEED == 1111
        assert c.SOLVER.IMS_PER_BATCH == 40
        assert c.SOLVER.MAX_EPOCHS == 120
        assert c.SOLVER.TRAIN_EPOCHS == 0
        c.freeze()
        print('     OK {}  HS={} FACSS={} QAWF={}'.format(
            row, actual[0], actual[1], actual[2]))


def _selection_inputs(dim=16, tokens=8, layers=3):
    features = [torch.randn(BATCH, tokens + 1, dim) for _ in range(3)]
    attentions = []
    for _ in range(3):
        modal_attn = []
        for _ in range(layers):
            logits = torch.randn(BATCH, 2, tokens + 1, tokens + 1)
            modal_attn.append(torch.softmax(logits, dim=-1))
        attentions.append(modal_attn)
    return features, attentions


def _run_selector(selector, features, attentions):
    return selector(
        RGB_feat=features[0], RGB_attn=attentions[0],
        NIR_feat=features[1], NIR_attn=attentions[1],
        TIR_feat=features[2], TIR_attn=attentions[2],
    )


def test_hs_facss_modes():
    print('[10] explicit M0 / HS-only / HS+FACSS paths')
    features, attentions = _selection_inputs()

    c = default_cfg.clone()
    c.MODEL.HS_ENABLED = 0
    c.MODEL.FACSS_ENABLED = 0
    bypass = HSFACSS(dim=16, cfg=c).eval()
    bypass_out = _run_selector(bypass, features, attentions)
    for original, selected in zip(features, bypass_out[:3]):
        assert torch.equal(original, selected)
    for mask in bypass_out[3]:
        assert mask.all()
    assert not hasattr(bypass, 'alpha_mlp')
    assert not hasattr(bypass, 'k_mlp')
    print('     OK M0 bypasses token selection')

    c.MODEL.HS_ENABLED = 1
    c.MODEL.FACSS_ENABLED = 0
    c.MODEL.HS_LAYERS = [1, 2, 3]
    c.MODEL.HS_K = 2
    hs_only = HSFACSS(dim=16, cfg=c).eval()
    hs_out = _run_selector(hs_only, features, attentions)
    assert not hasattr(hs_only, 'alpha_mlp')
    assert not hasattr(hs_only, 'k_mlp')
    for selected, mask in zip(hs_out[:3], hs_out[3]):
        assert (mask.sum(dim=1) >= 2).all()
        assert (mask.sum(dim=1) <= 6).all()
        dropped = selected[:, 1:, :][~mask]
        assert torch.count_nonzero(dropped) == 0
    print('     OK M1 runs HS without FACSS parameters or cross-modal refinement')

    c.MODEL.FACSS_ENABLED = 1
    c.MODEL.HS_LAYERS = [1]
    c.MODEL.HS_K = 8
    c.MODEL.FACSS_DYNAMIC_K = 0
    c.MODEL.FACSS_K = 2
    c.MODEL.FACSS_SOFT_RESIDUAL_WEIGHT = 0.0
    c.MODEL.FACSS_STE = 0
    c.MODEL.FACSS_MODALITY_UNION = 0
    facss = HSFACSS(dim=16, cfg=c).eval()
    facss_out = _run_selector(facss, features, attentions)
    assert hasattr(facss, 'alpha_mlp')
    assert hasattr(facss, 'k_mlp')
    for mask in facss_out[3]:
        assert torch.equal(mask.sum(dim=1), torch.full((BATCH,), 2))
    print('     OK M2 applies FACSS fixed-K refinement on HS candidates')

    c.MODEL.HS_ENABLED = 0
    try:
        HSFACSS(dim=16, cfg=c)
    except ValueError as exc:
        assert 'requires MODEL.HS_ENABLED' in str(exc)
    else:
        raise AssertionError('FACSS without HS should fail configuration validation')
    print('     OK invalid FACSS-without-HS configuration is rejected')


def test_paper_model_modes():
    print('[11] paper M0-M3 end-to-end train/eval smoke')
    for row in PAPER_ROWS:
        c = _make_paper_cfg(row)
        c.MODEL.PRETRAIN_CHOICE = 'self'
        c.MODEL.SIE_CAMERA = False
        c.INPUT.SIZE_TRAIN = [128, 64]
        c.INPUT.SIZE_TEST = [128, 64]
        model = make_model(c, num_class=NUM_CLASSES, camera_num=0).cpu()

        def frequency_must_not_run(*args, **kwargs):
            raise AssertionError('paper configs must bypass the frequency branch')

        model.FREQ_INDEX.forward = frequency_must_not_run
        model.train()
        x = _dummy_batch(c)
        label = torch.randint(0, NUM_CLASSES, (BATCH,))
        output = model(x, cam_label=None, label=label, epoch=0)
        assert len(output) == 5, '{} expected 5 outputs, got {}'.format(row, len(output))
        for i, value in enumerate(output):
            _assert_finite(value, '{} output[{}]'.format(row, i))
        _loss_assembly_like_processor(output).backward()

        model.eval()
        with torch.no_grad():
            descriptor = model(x, cam_label=None, epoch=0)
        assert descriptor.shape == (BATCH, 3 * model.BACKBONE.token_dim)
        _assert_finite(descriptor, '{} descriptor'.format(row))
        print('     OK {} train+bwd+eval'.format(row))


def main():
    test_defaults_load()
    test_yml_configs_merge()
    test_iteration_scheduler()
    for y in YMLS:
        test_three_modal_pipeline(y)
    test_two_modal_pipeline()
    test_part_descriptor_mode()
    test_save_load_roundtrip()
    test_ablation_switches()
    test_paper_config_matrix()
    test_hs_facss_modes()
    test_paper_model_modes()
    print('\n=== ALL PIPELINE TESTS PASSED ===')


if __name__ == '__main__':
    main()
