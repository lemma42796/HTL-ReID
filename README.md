# HTL-ReID

**Hierarchical Token Learning with Dynamic Heterogeneous Fusion for Multi-Modal Object Re-Identification**

HTL-ReID is a Transformer-based framework for RGB/NIR/TIR object re-identification. The model is currently under controlled re-evaluation; verified results are recorded in `实验记录/`.

## Method

The research line combines hierarchical token selection, adaptive cross-modal interaction, part-aware representation, and multi-route fusion. The final architecture will be documented after the clean comparison is complete; historical values must be checked against their E-records.

## Installation

```bash
pip install -r requirements.txt
```

The project uses PyTorch and a ViT-B/16 backbone.

## Data Preparation

Set `DATASETS.ROOT_DIR` to the dataset root. RGBNT201, RGBNT100, and MSVR310 must follow their official directory organization.

## Training

Configuration files can be chained from left to right. Select the dataset base and matching HTL-ReID fusion configuration under `configs/`:

```bash
python train_net.py \
  --config_file /path/to/base.yml \
  --config_file /path/to/fusion.yml \
  DATASETS.ROOT_DIR /path/to/datasets \
  MODEL.PRETRAIN_PATH_T /path/to/vit_base_patch16_224.pth \
  OUTPUT_DIR /path/to/output
```

## Evaluation

```bash
python test_net.py \
  --config_file /path/to/base.yml \
  --config_file /path/to/fusion.yml \
  --config_file /path/to/evaluation.yml \
  DATASETS.ROOT_DIR /path/to/datasets \
  TEST.WEIGHT /path/to/checkpoint.pth
```

The main evaluation protocol disables re-ranking. Reproducible experiment details are indexed in [`实验记录.md`](实验记录.md).

## License

This project is released under the terms of the [LICENSE](LICENSE) file.
