# HTL-ReID

HTL-ReID is a research codebase for robust multi-modal object re-identification
with RGB / NIR / TIR inputs.

## Mainline

The paper path is evaluated through four controlled RGBNT201 rows: M0 shared
ViT-B/16, M1 adds hierarchical token selection (HS), M2 adds FACSS token
filtering, and M3 adds quality-aware selection/fusion weights (QAWF).

The shared protocol is `configs/RGBNT201/paper/base.yml`; merge exactly one of
`m0.yml` through `m3.yml` from the same directory. The older A0-A5 chain20
overlays are archived diagnostic configurations, not formal paper configs.

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

## Smoke Test

```bash
python test_pipeline.py
```

The smoke test checks config merging, scheduler semantics, 3-modal and 2-modal
forward/backward passes, save/load, explicit HS/FACSS switches, and all four
paper rows without real datasets.

## License

This project is released under the terms of the [LICENSE](LICENSE) file in this repository.
