# HTL-ReID

HTL-ReID is a research codebase for robust multi-modal object re-identification
with RGB / NIR / TIR inputs.

## Mainline

The current evidence-backed path is A2: shared ViT-B/16, hierarchical token
selection (HS), FACSS token filtering, and quality-aware selection/fusion
weights (QAWF).

A3-A5 overlays, AGF/TPM fusion, modality adapters, part branches, and auxiliary
loss/full branches are kept only for archived ablations or negative evidence.
They are not the current paper-method path. HSL is not part of this model path.

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
python train_net.py --config_file configs/RGBNT201/default.yml \
    --config_file configs/RGBNT201/ablations/chain20/a2_quality.yml \
    DATASETS.ROOT_DIR /path/to/datasets \
    MODEL.PRETRAIN_PATH_T /path/to/pretrained_vit.pth
```

The A2 overlay sets `SOLVER.TRAIN_EPOCHS: 20` while keeping
`SOLVER.MAX_EPOCHS: 120`, so the cosine schedule still follows the 120-epoch
contract.

## Evaluation

```bash
python test_net.py --config_file configs/RGBNT201/default.yml \
    --config_file configs/RGBNT201/ablations/chain20/a2_quality.yml \
    TEST.WEIGHT /path/to/checkpoint.pth
```

`TEST.RE_RANKING` is enabled by default in the dataset configs.

## Smoke Test

```bash
python test_pipeline.py
```

The smoke test checks config merging, scheduler semantics, 3-modal and 2-modal
forward/backward passes, save/load, and ablation switches without real datasets.

## License

This project is released under the terms of the [LICENSE](LICENSE) file in this repository.
