# HTL-ReID Old-Method Recovery

This branch restores and evaluates the submitted-method line:

```text
Shared ViT-B/16 → HS → FACSS → AGF
```

It is intentionally separate from the revised `HS–ACI–PLR–DHF` model on the
main branch.

## Result

On RGBNT201 with batch 64 and no re-ranking, flip TTA, or ensemble, the best
recovered old-method result is **66.89% mAP / 69.02% Rank-1**. It uses the
structural-rescue checkpoint with AGF descriptor interpolation weight `0.4`.
Further old-method tuning is stopped; this branch is retained as implementation
and reviewer-response evidence.

## Recovery configurations

- `configs/RGBNT201/recovery/base.yml`: common protocol.
- `r001_backbone.yml`: backbone baseline.
- `r002_hs.yml`: HS.
- `r003_hs_facss.yml`: paper-style FACSS.
- `r004_full.yml`: paper-style cascaded AGF.
- `r004_structural.yml`: differentiable FACSS slots, local supervision, and
  active residual AGF.
- `r004_structural_eval_w04.yml`: best evaluation interpolation overlay.

## Training

All model execution must run on a CUDA machine. Multiple configuration files
are merged from left to right:

```bash
python train_net.py \
  --config_file configs/RGBNT201/recovery/base.yml \
  --config_file configs/RGBNT201/recovery/r004_structural.yml
```

## Evaluation

```bash
python test_net.py \
  --config_file configs/RGBNT201/recovery/base.yml \
  --config_file configs/RGBNT201/recovery/r004_structural.yml \
  --config_file configs/RGBNT201/recovery/r004_structural_eval_w04.yml \
  TEST.WEIGHT /path/to/HTL-ReID-recovery_best.pth
```

## CUDA smoke test

```bash
python tools/test_recovery_gpu.py
```

The smoke test checks every recovery row, three-modal forward/backward,
descriptor shape, finite outputs, and gradients through FACSS and AGF.

## License

This project is released under the terms of the [LICENSE](LICENSE) file.
