# HTL-ReID

Official implementation of the manuscript "**Hierarchical Token Learning and Adaptive Gated Fusion for Robust Multi-Modal Object Re-Identification**".

## Current Evidence-Backed Path

The current maintained chapter-3 path is the A2 configuration: shared ViT-B/16
with hierarchical token selection, FACSS, and quality-aware selection/fusion
weights. Use `configs/RGBNT201/ablations/chain20/a2_quality.yml` as the
recommended RGBNT201 evidence-backed overlay.

The repository still keeps later AGF/TPM fusion, modality-adapter, part-branch,
and auxiliary-loss switches for ablation and follow-up research. The latest
internal chain20 checks did not show a stable mAP gain from A3 over A2, so those
branches should be treated as exploratory unless they are revalidated.

## Overview

HTL-ReID is a unified framework for nighttime multi-modal (RGB / NIR / TIR) object re-identification. The current evidence-backed implementation extends the original HTL design into a quality-aware token selection and fusion pipeline:

- **Hierarchical Token Selection (HS)** — aggregates attention cues from shallow, middle, and deep ViT layers as complementary spatial priors, while keeping all token features in a single deep semantic space.
- **Dynamic FACSS** — jointly scores intra-modal discriminability and quality-weighted cross-modal cosine consensus, predicts a per-sample/per-modality token budget, and keeps a soft residual path outside hard top-k selection.
- **Quality-aware frequency selection** — uses predicted modality reliability to weight RGB / NIR / TIR wavelet cues before frequency tokens enter the FACSS candidate set.
- **Nighttime modality quality estimation** — predicts RGB / NIR / TIR reliability with a night-scene prior so weak RGB, overexposed NIR, or noisy TIR evidence can be down-weighted per sample.
- **Exploratory fusion branches** — AGF/TPM-style fusion, modality adapters, local part descriptors, and cross-modal auxiliary constraints are available as research switches, but are not the recommended mainline at the current evidence checkpoint.

HSL is not part of the current model path.

## Requirements
```bash
pip install -r requirements.txt
```

> `pytorch_wavelets` is vendored under `./pytorch_wavelets` — do **not** `pip install` it separately.

## Datasets
Download **RGBNT201**, **RGBNT100**, and **MSVR310** from their official sources, then update `DATASETS.ROOT_DIR` in the corresponding YAML config under `./configs/` to point to your local dataset directory.

## Pretrained Backbone
Download a pretrained ViT checkpoint and set `MODEL.PRETRAIN_PATH_T` in the YAML config to its local path. The backbone variant is selected via `MODEL.TRANSFORMER_TYPE`; supported values are listed in `modeling/make_model.py`.

## Training
Train on a single dataset by selecting its YAML config:

```bash
# RGBNT201
python train_net.py --config_file configs/RGBNT201/default.yml

# RGBNT100
python train_net.py --config_file configs/RGBNT100/default.yml

# MSVR310
python train_net.py --config_file configs/MSVR310/default.yml
```

For the current RGBNT201 chapter-3 evidence path, merge the dataset default with
the A2 overlay:

```bash
python train_net.py --config_file configs/RGBNT201/default.yml \
    --config_file configs/RGBNT201/ablations/chain20/a2_quality.yml
```

`configs/RGBNT201/ablations/chain20/a3_agf.yml` is kept for exploratory AGF/TPM
checks and should not be treated as the default final configuration without a
fresh validation run.

The dataset defaults expose the full feature surface: AdamW, a smaller learning rate for the shared ViT backbone, higher-resolution inputs, grayscale patch replacement, modality dropout, quality-aware frequency selection, ID + triplet supervision, optional part-branch supervision, staged branch/auxiliary loss weighting, and optional cross-modal auxiliary losses.
`SOLVER.WARMUP_ITERS` is update-based by default via `SOLVER.SCHEDULER_UNIT: iteration`; set the unit to `epoch` only when you intentionally want epoch-level scheduling.

You can override any config field from the command line, e.g.:
```bash
python train_net.py --config_file configs/RGBNT201/default.yml \
    DATASETS.ROOT_DIR /path/to/datasets \
    MODEL.PRETRAIN_PATH_T /path/to/pretrained_vit.pth
```

## Evaluation
```bash
python test_net.py --config_file configs/RGBNT201/default.yml \
    TEST.WEIGHT /path/to/checkpoint.pth
```

`TEST.RE_RANKING` is enabled by default in the dataset YAML files for final evaluation.
When a part-branch checkpoint is used, `TEST.PART_FEAT concat` appends a normalized part descriptor to the fused descriptor. Set `TEST.PART_FEAT off` to recover the fused descriptor alone.

## Smoke Test
Run the CPU-only pipeline test after installation or code changes:

```bash
python test_pipeline.py
```

The smoke test checks config merging, iteration scheduler semantics, 3-modal and 2-modal forward/backward passes, optional part descriptors, save/load round-trip, and ablation switches without real datasets or pretrained weights.

## License
This project is released under the terms of the [LICENSE](LICENSE) file in this repository.
