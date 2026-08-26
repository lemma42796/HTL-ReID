# HTL-ReID

## Project documentation

- [Project status and TODO](项目状态与TODO.md)
- [Paper revision plan](论文大修执行方案.md)
- [Experiment index](实验记录.md)
- [Reproducible experiment records](实验记录/)

HTL-ReID is a research codebase for robust multi-modal object re-identification
with RGB / NIR / TIR inputs.

## Mainline

The paper path is evaluated through four controlled RGBNT201 rows: M0 shared
ViT-B/16, M1 adds hierarchical token selection (HS), M2 adds FACSS token
filtering, and M3 adds quality-aware selection/fusion weights (QAWF).

The shared protocol is `configs/RGBNT201/paper/base.yml`; merge exactly one of
`m0.yml` through `m3.yml` from the same directory. The older A0-A5 chain20
overlays are archived diagnostic configurations, not formal paper configs.

The old AGF-wrapped TPM path remains archived negative evidence. A separate,
clean TOP-ReID TPM reproduction and the FACSS-guided FACR extension live under
`configs/RGBNT201/fusion`; neither changes the completed M0-M3 results.
Modality adapters, part branches, and auxiliary loss/full branches are not the
current paper-method path. HSL is not part of this model path.

## Cross-modal fusion extension

Merge the frozen paper base with exactly one fusion overlay:

- `t1_tpm.yml`: attributed TPM reproduction on complete backbone tokens.
- `t2_adaptive_routing.yml`: adaptive all-connected routing without FACSS.
- `t3_m2_facr.yml`: dense FACSS scores softly guide adaptive routing.

TPM/FACR directly produce the supervised 3D descriptor. They do not use the
legacy AGF wrapper, hard-pruned fusion input, or a `0.15` auxiliary concat.

## Requirements

```bash
pip install -r requirements.txt
```

`pytorch_wavelets` is vendored under `./pytorch_wavelets`; do not install it
separately.

Set `DATASETS.ROOT_DIR` and `MODEL.PRETRAIN_PATH_T` in the dataset config, or
override them from the command line.

## Training

```bash
python train_net.py --config_file configs/RGBNT201/paper/base.yml \
    --config_file configs/RGBNT201/paper/m2.yml \
    DATASETS.ROOT_DIR /path/to/datasets \
    MODEL.PRETRAIN_PATH_T /path/to/pretrained_vit.pth
```

The controlled paper protocol trains for 20 epochs while retaining the
120-epoch cosine-schedule horizon. It uses batch size 40, seed 1111, disables
periodic checkpoints while retaining the best checkpoint, and must be launched
with a 30-minute wall-clock timeout.

## Evaluation

```bash
python test_net.py --config_file configs/RGBNT201/paper/base.yml \
    --config_file configs/RGBNT201/paper/m2.yml \
    TEST.WEIGHT /path/to/checkpoint.pth
```

The paper configs explicitly disable re-ranking for the main results.

## Legacy-style A2 comparison

The current implementation can also reproduce the old A2 module combination:
HS + FACSS + quality weighting + quality-aware frequency selection. This is a
comparison with the old code path, not a fifth paper ablation row. Merge the
legacy overlay after the frozen paper base:

```bash
timeout --signal=TERM --kill-after=10s 30m \
  /root/miniconda3/bin/python train_net.py \
  --config_file configs/RGBNT201/paper/base.yml \
  --config_file configs/RGBNT201/legacy/a2_quality_frequency.yml \
  OUTPUT_DIR /root/autodl-tmp/outputs/HTL-ReID/E005_L1_legacy_a2_seed1111
```

This uses RGBNT201, seed 1111, batch size 40, and 20 epochs. The base config
keeps re-ranking off, so evaluate the saved best checkpoint once with the same
two configs for the primary comparison. For a separately labeled old-protocol
number, append
`--config_file configs/RGBNT201/legacy/eval_rerank.yml` to the evaluation
command. Both evaluations reuse the same checkpoint; do not retrain.

## Remaining controlled rows

After E001/M0 has completed, run M1-M3 sequentially with independent
initialization and a separate 30-minute cap per row:

```bash
python tools/run_rgbnt201_paper_remaining.py
```

Use `--dry-run` to inspect the experiment IDs, configs, output directories,
epoch count, seed, and timeout without creating artifacts or starting training.

## Smoke Test

For the new fusion rows, use the CUDA-only smoke test (it intentionally has no
CPU fallback):

```bash
python tools/smoke_fusion_gpu.py
```

The broader legacy/paper regression suite remains available separately:

```bash
python test_pipeline.py
```

The smoke test checks config merging, scheduler semantics, 3-modal and 2-modal
forward/backward passes, save/load, explicit HS/FACSS switches, all four paper
rows, and the legacy-style quality-aware frequency path without real datasets.

## License

This project is released under the terms of the [LICENSE](LICENSE) file in this repository.
