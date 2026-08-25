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
 16. The legacy-style A2 config runs quality-aware frequency selection
 17. Optimized rollout, cross-score, dynamic-K, and frequency kernels preserve
     their reference outputs
 18. AdamW uses grouped multi-tensor parameter groups

Run:
    python3 test_pipeline.py
"""
import copy
import io
import sys
import torch

from config import cfg as default_cfg
from modeling.make_model import make_model
from modeling.fusion_part.HS_FACSS import HSFACSS, HierarchicalRollout
from modeling.fusion_part.Frequency import Frequency_based_Token_Selection
from modeling.fusion_part.TPM import ScoreBiasedCrossAttention
from solver.make_optimizer import make_optimizer
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
LEGACY_A2 = 'configs/RGBNT201/legacy/a2_quality_frequency.yml'
LEGACY_RERANK = 'configs/RGBNT201/legacy/eval_rerank.yml'

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


def _make_legacy_a2_cfg():
    c = default_cfg.clone()
    c.merge_from_file(PAPER_BASE)
    c.merge_from_file(LEGACY_A2)
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
        assert not c.MODEL.TPM
        assert not c.MODEL.FACR
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
        assert c.SOLVER.TRAIN_EPOCHS == 20
        assert c.SOLVER.EVAL_PERIOD == 5
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


def test_optimized_kernels_equivalent():
    print('[11] optimized selection kernels match reference implementations')
    torch.manual_seed(7)

    layers = (1, 2, 4)
    attn_list = [
        torch.softmax(torch.randn(3, 2, 9, 9), dim=-1)
        for _ in range(max(layers))
    ]
    rollout = HierarchicalRollout(layers=layers, k=3)
    mask_new, final_new, snapshots_new = rollout(attn_list)

    eye = torch.eye(9).unsqueeze(0)
    cumulative = None
    snapshots_ref = {}
    for layer, attn in enumerate(attn_list, start=1):
        transition = 0.5 * attn.mean(dim=1) + 0.5 * eye
        cumulative = transition if cumulative is None else transition @ cumulative
        if layer in layers:
            snapshots_ref[layer] = cumulative[:, 0, 1:]
    mask_ref = torch.zeros_like(mask_new)
    for layer in layers:
        indices = snapshots_ref[layer].topk(3, dim=1).indices
        mask_ref.scatter_(1, indices, True)
    assert torch.equal(mask_new, mask_ref)
    assert torch.allclose(final_new, snapshots_ref[max(layers)], atol=1e-6, rtol=1e-5)
    for layer in layers:
        assert torch.allclose(snapshots_new[layer], snapshots_ref[layer], atol=1e-6, rtol=1e-5)

    c = default_cfg.clone()
    c.MODEL.HS_LAYERS = [1]
    c.MODEL.HS_K = 8
    c.MODEL.FACSS_DYNAMIC_K = 1
    c.MODEL.FACSS_MIN_K = 1
    c.MODEL.FACSS_MAX_K = 3
    c.MODEL.FACSS_SOFT_RESIDUAL_WEIGHT = 0.0
    c.MODEL.FACSS_STE = 0
    c.MODEL.FACSS_MODALITY_UNION = 0
    selector = HSFACSS(dim=16, cfg=c).eval()
    features, attentions = _selection_inputs(dim=16, tokens=8, layers=1)
    selected = _run_selector(selector, features, attentions)
    for mask in selected[3]:
        assert torch.equal(mask.sum(dim=1), torch.full((BATCH,), 2))

    patches = {
        'RGB': features[0][:, 1:],
        'NIR': features[1][:, 1:],
        'TIR': features[2][:, 1:],
    }
    quality = {
        'RGB': torch.tensor([0.3, 0.7]),
        'NIR': torch.tensor([0.5, 0.4]),
        'TIR': torch.tensor([0.8, 0.6]),
    }
    cached = selector._cross_scores(patches, quality=quality)
    for name in patches:
        other_names = [other for other in patches if other != name]
        reference_parts = []
        for other in other_names:
            left = torch.nn.functional.normalize(patches[name], dim=-1)
            right = torch.nn.functional.normalize(patches[other], dim=-1)
            cosine = (left @ right.transpose(-1, -2)).relu()
            reference_parts.append(selector._pool_cross(cosine))
        reference_parts = torch.stack(reference_parts, dim=1)
        weights = torch.stack([quality[other] for other in other_names], dim=1).unsqueeze(-1)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
        reference = (reference_parts * weights).sum(dim=1)
        assert torch.allclose(cached[name], reference, atol=1e-6, rtol=1e-5)

    frequency = Frequency_based_Token_Selection(keep=3, stride=4)
    inverse = torch.randn(5, 3, 16, 12)
    mask_fast = frequency.mask(inverse, window_size=4)
    counts = []
    grayscale = inverse.mean(dim=1)
    for sample in grayscale:
        unfolded = torch.nn.functional.unfold(
            sample[None, None], kernel_size=4, stride=4)
        counts.append(unfolded.gt(0).sum(dim=1).squeeze(0))
    counts = torch.stack(counts)
    idx = counts.topk(3, dim=1).indices
    mask_ref = torch.zeros_like(mask_fast)
    mask_ref.scatter_(1, idx, True)
    assert torch.equal(mask_fast, mask_ref)
    print('     OK rollout, cached cosine, dynamic-K, and frequency mask')


def test_optimizer_parameter_groups():
    print('[12] optimizer groups tensors by effective hyperparameters')

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.BACKBONE = torch.nn.Linear(4, 4)
            self.FUSE_HEAD = torch.nn.Linear(4, 2)
            self.extra_norm = torch.nn.LayerNorm(4)

    c = default_cfg.clone()
    model = TinyModel()
    center = torch.nn.Linear(2, 2)
    optimizer, _ = make_optimizer(c, model, center)
    tensor_count = sum(1 for parameter in model.parameters() if parameter.requires_grad)
    grouped_count = len(optimizer.param_groups)
    optimized_count = sum(len(group['params']) for group in optimizer.param_groups)
    assert optimized_count == tensor_count
    assert grouped_count < tensor_count
    assert optimizer.defaults['foreach'] is True
    print('     OK {} tensors -> {} optimizer groups'.format(tensor_count, grouped_count))


def test_facr_score_bias_starts_from_t2():
    print('[13] FACR score bias and hard mask behavior')
    torch.manual_seed(11)
    attention = ScoreBiasedCrossAttention(
        dim=16, num_heads=2, score_bias_scale=0.25,
        detach_scores=False).eval()
    query = torch.randn(2, 16)
    source = torch.randn(2, 6, 16)
    scores = torch.rand(2, 6)

    without_scores = attention(query, source)
    with_scores = attention(query, source, scores=scores)
    assert torch.allclose(without_scores, with_scores, atol=1e-7, rtol=0.0)
    assert attention.score_bias_gain.item() == 0.0

    with torch.no_grad():
        attention.score_bias_gain.fill_(10.0)
    guided = attention(query, source, scores=scores)
    assert not torch.allclose(without_scores, guided)
    effective_gain = (
        attention.score_bias_scale * torch.tanh(attention.score_bias_gain)
    ).abs().item()
    assert effective_gain <= 0.25

    mask = torch.tensor([
        [True, True, True, False, False, False],
        [True, False, True, False, True, False],
    ])
    masked = attention(query, source, mask=mask)
    changed_source = source.clone()
    changed_source[~mask] = torch.randn_like(changed_source[~mask]) * 1e4
    changed_masked = attention(query, changed_source, mask=mask)
    assert torch.allclose(masked, changed_masked, atol=1e-6, rtol=1e-5)
    try:
        attention(query, source, mask=torch.zeros_like(mask))
    except ValueError as exc:
        assert 'at least one' in str(exc)
    else:
        raise AssertionError('an empty FACSS mask must be rejected')

    t4 = default_cfg.clone()
    t4.merge_from_file(PAPER_BASE)
    t4.merge_from_file('configs/RGBNT201/fusion/t4_facss_masked_facr.yml')
    assert t4.MODEL.FACR
    assert t4.MODEL.FACR_USE_MASKS
    assert not t4.MODEL.FACR_USE_SCORES
    assert not t4.MODEL.FACSS_DYNAMIC_K
    assert t4.MODEL.FACSS_K == 16
    print('     OK initial output=T2; gain bounded at {:.3f}; mask is hard'.format(
        effective_gain))


def test_paper_model_modes():
    print('[14] paper M0-M3 end-to-end train/eval smoke')
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


def test_legacy_a2_quality_frequency():
    print('[15] legacy-style A2 quality-aware frequency path')
    c = _make_legacy_a2_cfg()
    assert c.MODEL.HS_ENABLED
    assert c.MODEL.FACSS_ENABLED
    assert c.MODEL.QUALITY_AWARE
    assert c.MODEL.FREQUENCY_ENABLED
    assert c.MODEL.FREQUENCY_QUALITY_AWARE
    assert c.MODEL.FREQUENCY_KEEP == 10
    assert not c.MODEL.AGF
    assert not c.MODEL.MODALITY_ADAPTER
    assert not c.MODEL.PART_BRANCH
    assert not c.MODEL.OCFR
    assert c.TEST.RE_RANKING == 'no'
    assert c.SOLVER.SEED == 1111
    assert c.SOLVER.IMS_PER_BATCH == 40
    assert c.SOLVER.TRAIN_EPOCHS == 20
    rerank_cfg = c.clone()
    rerank_cfg.merge_from_file(LEGACY_RERANK)
    assert rerank_cfg.TEST.RE_RANKING == 'yes'

    c.MODEL.PRETRAIN_CHOICE = 'self'
    c.MODEL.SIE_CAMERA = False
    c.INPUT.SIZE_TRAIN = [128, 64]
    c.INPUT.SIZE_TEST = [128, 64]
    model = make_model(c, num_class=NUM_CLASSES, camera_num=0).cpu()
    assert model.use_frequency
    assert model.use_quality
    assert model.FREQ_INDEX.quality_aware

    frequency_calls = []
    original_frequency_forward = model.FREQ_INDEX.forward

    def track_frequency_call(*args, **kwargs):
        quality_scores = kwargs.get('quality_scores')
        assert quality_scores is not None
        assert quality_scores.shape == (BATCH, 3)
        frequency_calls.append(quality_scores.detach().clone())
        return original_frequency_forward(*args, **kwargs)

    model.FREQ_INDEX.forward = track_frequency_call
    model.train()
    x = _dummy_batch(c)
    label = torch.randint(0, NUM_CLASSES, (BATCH,))
    output = model(x, cam_label=None, label=label, epoch=0)
    assert len(output) == 5
    for i, value in enumerate(output):
        _assert_finite(value, 'legacy A2 output[{}]'.format(i))
    _loss_assembly_like_processor(output).backward()

    model.eval()
    with torch.no_grad():
        descriptor = model(x, cam_label=None, epoch=0)
    assert descriptor.shape == (BATCH, 3 * model.BACKBONE.token_dim)
    _assert_finite(descriptor, 'legacy A2 descriptor')
    assert len(frequency_calls) == 2
    print('     OK train+bwd+eval; frequency received learned quality scores')


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
    test_optimized_kernels_equivalent()
    test_optimizer_parameter_groups()
    test_facr_score_bias_starts_from_t2()
    test_paper_model_modes()
    test_legacy_a2_quality_frequency()
    print('\n=== ALL PIPELINE TESTS PASSED ===')


if __name__ == '__main__':
    main()
