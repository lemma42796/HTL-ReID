from yacs.config import CfgNode as CN

_C = CN()
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Name of model
_C.MODEL.NAME = 'HTL-ReID'
# Margin of triplet loss
_C.MODEL.MARGIN = 0
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH_T = '/root/autodl-tmp/pretrained/vit_base_patch16_224_augreg2_in21k_ft_in1k.pth'
# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' or 'self'
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'
# Optional full-model checkpoint to partially load before training.
_C.MODEL.RESUME_PATH = ''
# Choose different resnet backbone (1->resnet50||2->resnet101||3->resnet152)
# Fusion dim
_C.MODEL.MIX_DIM = 768
# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 'bnneck'
# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C.MODEL.IF_WITH_CENTER = 'no'
_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
# The loss type of metric loss
# options:['triplet'](without center loss) or ['center','triplet_center'](with center loss)
_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
# If train with multi-gpu ddp mode, options: 'True', 'False'
_C.MODEL.DIST_TRAIN = False
# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'
# Choose the supervision type of the backbone
_C.MODEL.AL = 0
# [Deprecated] kept for backward-compatible cfg loading; replaced by HS_K
_C.MODEL.HEAD_KEEP = 1
# The keep tokens in the Frequency Selection Part
_C.MODEL.FREQUENCY_KEEP=10
_C.MODEL.FREQUENCY_QUALITY_AWARE = 1
_C.MODEL.FREQUENCY_ENABLED = 1
# HS (Hierarchical Token Selection): cross-layer attention rollout plus the
# frequency mask, merged into one shared selection mask. Selection rule from
# the official EDITOR source.
_C.MODEL.HS_ENABLED = 0
_C.MODEL.HS_RATIO = 0.5
# Learn a dataset-level discrete tokens-per-head K with a straight-through
# Gumbel-Softmax architecture choice. The final union density is regularized
# so identification loss cannot trivially collapse the choice to maximum K.
_C.MODEL.HS_LEARNABLE_K = 0
_C.MODEL.HS_K_CANDIDATES = [1, 2, 4, 8, 16]
_C.MODEL.HS_GUMBEL_TAU = 1.0
_C.MODEL.HS_GUMBEL_TAU_MIN = 0.2
_C.MODEL.HS_GUMBEL_TAU_DECAY = 0.9
_C.MODEL.HS_BUDGET_LOSS_WEIGHT = 0.05
# Preserve information outside the shared hard mask as one attention-weighted
# residual token per modality. Consumed only by FACR final self-refinement.
_C.MODEL.HS_RESIDUAL_TOKEN = 0
_C.MODEL.SELECTED_PATCH_BLEND_WEIGHT = 0.15
_C.MODEL.SELECTED_PATCH_CONTEXT = 'mean'
_C.MODEL.SELECTED_PATCH_ATTN_SCALE = 10.0
_C.MODEL.SELECTED_PATCH_GATE_INIT = 0.0
_C.MODEL.SELECTED_AGGREGATION = 0
_C.MODEL.SELECTED_AGG_NUM_HEADS = 12
_C.MODEL.SELECTED_AGG_GATE_INIT_BIAS = -2.0
_C.MODEL.SELECTED_AGG_RESIDUAL_WEIGHT = 0.5
# FACSS-guided Adaptive Cross-modal Routing (project extension).
_C.MODEL.FACR = 0
_C.MODEL.FACR_USE_SCORES = 1
_C.MODEL.FACR_USE_MASKS = 0
_C.MODEL.FACR_NUM_HEADS = 12
_C.MODEL.FACR_STEPS = 3
_C.MODEL.FACR_SCORE_BIAS_SCALE = 0.25
_C.MODEL.FACR_SCORE_FLOOR = 0.05
_C.MODEL.FACR_DETACH_SCORES = 1
_C.MODEL.FACR_GATE_INIT_BIAS = 0.0
# Preserve the selector descriptor and add only FACR's routed delta. Existing
# experiments keep the historical replacement behavior unless explicitly
# enabled by a row config.
_C.MODEL.FACR_RESIDUAL_FUSION = 0
_C.MODEL.FACR_RESIDUAL_SCALE_INIT = 0.1
# Train FACR as an auxiliary branch from detached backbone tokens so its loss
# cannot alter the selector/main branch. The routed descriptor is supervised by
# its own BNNeck and classifier and can be concatenated only at test time.
_C.MODEL.FACR_ISOLATED_BRANCH = 0
# Optional own-modality masked aggregation before collaborative FACR stages.
# This follows the independent-before-collaborative ordering investigated by
# T11 while leaving every existing FACR configuration unchanged by default.
_C.MODEL.FACR_INDEPENDENT_AGG = 0
# Optional final own-modality patch read after the adaptive cross-modal stages.
_C.MODEL.FACR_SELF_REFINE = 0
_C.MODEL.FACR_SELF_REFINE_SCALE_INIT = 0.1
# Batch-level load balancing for FACR. It keeps both candidate source
# modalities used across a batch without forcing every sample to route 50/50.
_C.MODEL.FACR_ROUTE_BALANCE_WEIGHT = 0.0
# Training-only shared cross-modal token reconstruction. This is an attributed
# CRM-inspired auxiliary objective adapted to the shared-backbone setting; it
# never changes the inference descriptor or existing checkpoints by default.
_C.MODEL.CROSS_MODAL_RECON_ENABLED = 0
_C.MODEL.CROSS_MODAL_RECON_HIDDEN_DIM = 256
_C.MODEL.CROSS_MODAL_RECON_LOSS_WEIGHT = 0.1
_C.MODEL.CROSS_MODAL_RECON_ALL_TARGETS = 0
_C.MODEL.CROSS_MODAL_RECON_SMOOTH_L1_WEIGHT = 0.0
# Supervised batch-hard metric loss across different modality CLS features.
_C.MODEL.HETERO_TRIPLET_WEIGHT = 0.0
_C.MODEL.HETERO_TRIPLET_MARGIN = 0.3
# DeMo-inspired heterogeneous feature decomposition: three modality-specific,
# three pairwise-shared, and one all-modality route with dynamic expert gating.
_C.MODEL.DECOUPLED_MOE = 0
_C.MODEL.DECOUPLED_MOE_NUM_HEADS = 12
_C.MODEL.DECOUPLED_MOE_GATE_HEADS = 4
_C.MODEL.DECOUPLED_MOE_DROPOUT = 0.1
_C.MODEL.DECOUPLED_MOE_LOSS_WEIGHT = 1.0
# OCFR auxiliary loss (not in paper); off by default for paper-faithful reproduction
_C.MODEL.OCFR = 0
# Nighttime modality reliability. The prior only initializes the quality head:
# RGB is kept useful but starts slightly below NIR/TIR for night imagery.
_C.MODEL.QUALITY_AWARE = 1
_C.MODEL.QUALITY_HIDDEN = 192
_C.MODEL.QUALITY_PRIOR = [0.50, 0.65, 0.65]
_C.MODEL.QUALITY_MIN_SCORE = 0.05
# Lightweight modality adapters after the shared ViT backbone.
_C.MODEL.MODALITY_ADAPTER = 1
_C.MODEL.MODALITY_ADAPTER_DIM = 192
_C.MODEL.MODALITY_ADAPTER_SCALE = 0.25
# Local identity evidence from selected tokens.
_C.MODEL.PART_BRANCH = 1
_C.MODEL.PART_NUM = 3
# Part pooling: 'stripe' keeps the fixed horizontal pooling; 'semantic' learns
# selected-token part prototypes shared across modalities.
_C.MODEL.PART_POOL = 'stripe'
# Auxiliary cross-modal constraints.
_C.MODEL.ALIGN_LOSS_WEIGHT = 0.05
_C.MODEL.TOKEN_CONSISTENCY_WEIGHT = 0.01
_C.MODEL.BCC_LOSS_WEIGHT = 0.0
_C.MODEL.GATE_BALANCE_WEIGHT = 0.0
_C.MODEL.QUALITY_PERTURB_LOSS_WEIGHT = 0.01
_C.MODEL.FUSE_LOSS_WEIGHT = 1.0
_C.MODEL.BRANCH_LOSS_WEIGHT = 0.5
_C.MODEL.PART_LOSS_WEIGHT = 0.25
_C.MODEL.AUX_LOSS_WEIGHT = 0.3
_C.MODEL.AUX_WARMUP_EPOCHS = 40

# Transformer setting
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0
_C.MODEL.TRANSFORMER_TYPE = 'vit_base_patch16_224'
# The stride size of the backbone
_C.MODEL.STRIDE_SIZE = [16, 16]

# SIE Parameter
_C.MODEL.SIE_COE = 3.0
_C.MODEL.SIE_CAMERA = True
_C.MODEL.SIE_VIEW = False
# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [256, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [256, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]
# Value of padding size
_C.INPUT.PADDING = 10
_C.INPUT.GRAY_REPLACE_PROB = 0.3
_C.INPUT.MODALITY_DROP_PROB = 0.0

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('RGBNT201')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('/root/autodl-tmp/datasets')
# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 14
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax_triplet'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "AdamW"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 120
# Optional early stop for diagnostic/ablation runs. A value <= 0 means train
# through MAX_EPOCHS; the LR schedule still uses MAX_EPOCHS.
_C.SOLVER.TRAIN_EPOCHS = 0
# Base learning rate
_C.SOLVER.BASE_LR = 0.0001
_C.SOLVER.BACKBONE_LR_FACTOR = 0.1
_C.SOLVER.NEW_MODULE_LR_FACTOR = 1.0
_C.SOLVER.DECOUPLED_MOE_LR_FACTOR = 1.0
# Factor of learning bias
_C.SOLVER.LARGE_FC_LR = False
_C.SOLVER.BIAS_LR_FACTOR = 2
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.3
# Margin of cluster ;pss
_C.SOLVER.CLUSTER_MARGIN = 0.3
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005
# Settings of range loss
_C.SOLVER.RANGE_K = 2
_C.SOLVER.RANGE_MARGIN = 0.3
_C.SOLVER.RANGE_ALPHA = 0
_C.SOLVER.RANGE_BETA = 1
_C.SOLVER.RANGE_LOSS_WEIGHT = 1
# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.05
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0
# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# warm up factor
_C.SOLVER.WARMUP_FACTOR = 0.01
# iterations of warm up
_C.SOLVER.WARMUP_ITERS = 20
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"
# Scheduler step unit, options: 'iteration' or 'epoch'. Keep 'iteration' so
# WARMUP_ITERS is interpreted literally instead of as warmup epochs.
_C.SOLVER.SCHEDULER_UNIT = "iteration"

_C.SOLVER.COSINE_MARGIN = 0.5
_C.SOLVER.COSINE_SCALE = 30
_C.SOLVER.SEED = 1111
# Strict bitwise determinism: deterministic cuBLAS/cuDNN algorithms and
# torch.use_deterministic_algorithms. Seeded sampler/loader/reconstruction
# generators stay active regardless of this flag.
_C.SOLVER.STRICT_DETERMINISM = 1
_C.MODEL.NO_MARGIN = True
# epoch number of saving periodic model weights
_C.SOLVER.CHECKPOINT_PERIOD = 60
# Keep the best model weights for independent testing and evidence.
_C.SOLVER.SAVE_BEST_CHECKPOINT = True
# Optional single-run target checkpoint. Values are fractions in [0, 1].
_C.SOLVER.TARGET_MAP = 0.0
_C.SOLVER.TARGET_RANK1 = 0.0
# Periodic epoch weights are disabled by default to keep AutoDL disk usage low.
_C.SOLVER.SAVE_PERIODIC_CHECKPOINTS = False
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 10
# Validate periodically; processor.py always validates the final run epoch.
_C.SOLVER.EVAL_PERIOD = 5
_C.SOLVER.KL = 0
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 128 
# A batch size of 128 yields better results for both person and vehicle datasets compared to 64. 
# Using a batch size of 64 may result in a slight decrease in performance.

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 64
# If test with re-ranking, options: 'yes','no'
_C.TEST.RE_RANKING = 'yes'
# Path to trained model
_C.TEST.WEIGHT = ""
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'before'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'
# Optional test-time fusion with the trained part branch. 'concat' appends a
# normalized part descriptor; set to 'off' to use the fused descriptor alone.
_C.TEST.PART_FEAT = 'off'
_C.TEST.PART_FEAT_WEIGHT = 0.3
_C.TEST.ORIGINAL_CLS_FEAT = 'off'
_C.TEST.ORIGINAL_CLS_FEAT_WEIGHT = 0.5
_C.TEST.DECOUPLED_MOE_FEAT = 'off'
_C.TEST.DECOUPLED_MOE_FEAT_WEIGHT = 0.5
_C.TEST.FACR_ISOLATED_FEAT = 'off'
_C.TEST.FACR_ISOLATED_FEAT_WEIGHT = 0.25
# ----------------------------------------------------------a------------------ #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = "/root/autodl-tmp/outputs/HTL-ReID"
