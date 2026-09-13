# HTL-ReID

**Hierarchical Token Learning with Dynamic Heterogeneous Fusion for Multi-Modal Object Re-Identification**

HTL-ReID is a Transformer-based framework for multi-modal (RGB / NIR / TIR) object re-identification. Built on a shared ViT-B/16 backbone, it preserves cross-modal information flow through hierarchical token selection and gradient-isolated cross-modal interaction, and complements the global representation with part-aware local evidence.

## Highlights

- **Hierarchical token selection.** A shared-space selector keeps the most informative tokens across modalities, providing a compact common support for fusion.
- **Gradient-isolated cross-modal interaction.** Adaptive cross-modal routing refines features while its auxiliary branch is gradient-isolated from the backbone, avoiding the degradation caused by end-to-end coupling.
- **Part-aware representation.** A three-band part branch supplies local evidence on top of the global descriptor.
- **Robust under degradation.** The model degrades gracefully under missing-modality and occlusion conditions.

## Results

Main results under the standard protocol (fixed-epoch evaluation, no re-ranking, no test-time augmentation).

| Dataset  | mAP   | Rank-1 |
|----------|------:|-------:|
| RGBNT201 | 69.79 | 72.37  |
| RGBNT100 | 74.48 | 91.31  |
| MSVR310  | 33.24 | 48.73  |

## Installation

```bash
pip install -r requirements.txt
```

The project uses PyTorch and a ViT-B/16 backbone. Download the `vit_base_patch16_224` pretrained weights and reference them via `MODEL.PRETRAIN_PATH_T`.

## Data Preparation

Set `DATASETS.ROOT_DIR` to the dataset root. RGBNT201, RGBNT100, and MSVR310 must follow their official directory organization.

## Training

Configuration files are chained from left to right. Select the dataset base and the matching HTL-ReID fusion configuration under `configs/`:

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

The main evaluation protocol disables re-ranking.

## License

This project is released under the terms of the [LICENSE](LICENSE) file.
