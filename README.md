# HTL-ReID

**Hierarchical Token Learning with Dynamic Heterogeneous Fusion for Multi-Modal Object Re-Identification**

HTL-ReID is a Transformer-based framework for multi-modal object re-identification. It learns complementary representations from visible (RGB), near-infrared (NIR), and thermal infrared (TIR) images through hierarchical token selection and adaptive feature fusion.

## Highlights

- Hierarchical selection of informative cross-spectral tokens.
- Adaptive interaction between RGB, NIR, and TIR representations.
- Global and part-aware descriptors for fine-grained matching.
- Dynamic fusion of single-modal, bi-modal, and tri-modal routes.
- Support for RGBNT201, RGBNT100, and MSVR310.

## Method

HTL-ReID contains four main components:

- **HS — Hierarchical Token Selection:** combines multi-layer attention cues with frequency-aware masks to retain informative tokens.
- **ACI — Adaptive Cross-modal Interaction:** exchanges complementary evidence between modalities using sample-adaptive routing.
- **PLR — Part-aware Local Representation:** captures local discriminative patterns from horizontal regions.
- **DHF — Dynamic Heterogeneous Fusion:** aggregates single-modal, pairwise, and tri-modal routes with dynamic gates.

The final retrieval descriptor combines the cross-modal representation with global and dynamically fused features.

## Results

Representative performance on RGBNT201 without re-ranking:

| Dataset | mAP | Rank-1 |
|---|---:|---:|
| RGBNT201 | 77.71% | 82.66% |

## Installation

```bash
pip install -r requirements.txt
```

The project uses PyTorch and a ViT-B/16 backbone. `pytorch_wavelets` is included in the repository and does not need to be installed separately.

## Data Preparation

Set `DATASETS.ROOT_DIR` to a directory containing the target datasets:

```text
datasets/
├── RGBNT201/
│   ├── train_171/
│   └── test/
├── RGBNT100/
│   └── rgbir/
└── MSVR310/
    ├── bounding_box_train/
    ├── query3/
    └── bounding_box_test/
```

Each dataset should follow its official RGB/NIR/TIR organization.

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

Dataset-specific configurations for RGBNT100 and MSVR310 are provided under `configs/`.

## Evaluation

```bash
python test_net.py \
  --config_file /path/to/base.yml \
  --config_file /path/to/fusion.yml \
  --config_file /path/to/evaluation.yml \
  DATASETS.ROOT_DIR /path/to/datasets \
  TEST.WEIGHT /path/to/checkpoint.pth
```

The main evaluation protocol disables re-ranking. Descriptor analysis and evaluation utilities are available under `tools/`.

## Project Structure

```text
HTL-ReID/
├── configs/          # dataset and experiment configurations
├── data/             # datasets, samplers, and data loaders
├── modeling/         # backbone and HTL-ReID modules
├── layers/           # training objectives
├── engine/           # training and inference loops
├── tools/            # experiment and visualization utilities
├── train_net.py      # training entry point
└── test_net.py       # evaluation entry point
```

## License

This project is released under the terms of the [LICENSE](LICENSE) file.
