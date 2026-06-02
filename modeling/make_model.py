import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.backbones.vit_pytorch import vit_base_patch16_224, vit_small_patch16_224, \
    deit_small_patch16_224
from modeling.fusion_part.Frequency import Frequency_based_Token_Selection
from modeling.fusion_part.OCFR import OCFR
from modeling.fusion_part.HS_FACSS import HSFACSS
from modeling.fusion_part.AGF import AGF


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class ModalityQualityHead(nn.Module):
    """Predict nighttime modality reliability for RGB/NIR/TIR features."""

    def __init__(self, dim, hidden, prior, min_score):
        super().__init__()
        prior = torch.tensor(prior, dtype=torch.float32).clamp(1e-4, 1 - 1e-4)
        prior_logit = torch.log(prior / (1.0 - prior))
        self.register_buffer('prior_logit', prior_logit.view(1, 3))
        self.min_score = float(min_score)
        self.net = nn.Sequential(
            nn.LayerNorm(3 * dim),
            nn.Linear(3 * dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, rgb_cls, nir_cls, tir_cls=None):
        if tir_cls is None:
            tir_cls = torch.zeros_like(rgb_cls)
            available = torch.tensor([1.0, 1.0, 0.0], device=rgb_cls.device, dtype=rgb_cls.dtype)
        else:
            available = torch.ones(3, device=rgb_cls.device, dtype=rgb_cls.dtype)
        ctx = torch.cat([rgb_cls, nir_cls, tir_cls], dim=-1)
        prior_logit = self.prior_logit.to(device=ctx.device, dtype=ctx.dtype)
        score = torch.sigmoid(self.net(ctx) + prior_logit)
        score = self.min_score + (1.0 - self.min_score) * score
        return score * available.view(1, 3)


class ModalityAdapter(nn.Module):
    """Small residual adapter for modality-specific statistics after shared ViT."""

    def __init__(self, dim, hidden, scale):
        super().__init__()
        self.scale = float(scale)
        self.adapter = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, x):
        return x + self.scale * self.adapter(x)


class SelectedTokenAggregator(nn.Module):
    """HMA-like masked aggregation over selected patch tokens."""

    def __init__(self, dim, num_heads, gate_init_bias=-2.0, residual_weight=0.5):
        super().__init__()
        self.residual_weight = float(residual_weight)
        self.query_norm = nn.LayerNorm(dim)
        self.self_kv_norm = nn.LayerNorm(dim)
        self.cross_kv_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=0.0, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=0.0, batch_first=True)
        self.gate = nn.Linear(2 * dim, dim)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(gate_init_bias))

    @staticmethod
    def _prepare_tokens(feat, mask=None):
        patches = feat[:, 1:, :]
        if mask is None:
            valid = torch.ones(
                patches.shape[:2], device=patches.device, dtype=torch.bool)
        else:
            valid = mask.to(device=patches.device, dtype=torch.bool).clone()
            empty = ~valid.any(dim=1)
            if empty.any():
                valid[empty] = True
        return patches, ~valid

    def _attend(self, attn, kv_norm, query, tokens, key_padding_mask):
        q = self.query_norm(query).unsqueeze(1)
        kv = kv_norm(tokens)
        context, _ = attn(
            q, kv, tokens, key_padding_mask=key_padding_mask,
            need_weights=False)
        return context.squeeze(1)

    def forward(self, feats, masks=None, active=None, base_cls=None):
        masks = masks or (None, None, None)
        if len(masks) == 2:
            masks = (masks[0], masks[1], None)
        active = active or (True, True, True)
        base_cls = base_cls or [feat[:, 0, :] for feat in feats]

        prepared = [
            self._prepare_tokens(feat, mask)
            for feat, mask in zip(feats, masks)
        ]

        fused = []
        for target_idx, is_active in enumerate(active):
            base = base_cls[target_idx]
            if not is_active:
                fused.append(torch.zeros_like(base))
                continue

            own_tokens, own_padding = prepared[target_idx]
            self_ctx = self._attend(
                self.self_attn, self.self_kv_norm, base,
                own_tokens, own_padding)

            cross_tokens = []
            cross_padding = []
            for src_idx, src_active in enumerate(active):
                if src_idx == target_idx or not src_active:
                    continue
                tokens, padding = prepared[src_idx]
                cross_tokens.append(tokens)
                cross_padding.append(padding)

            if cross_tokens:
                tokens = torch.cat(cross_tokens, dim=1)
                padding = torch.cat(cross_padding, dim=1)
                cross_ctx = self._attend(
                    self.cross_attn, self.cross_kv_norm, base,
                    tokens, padding)
                context = 0.5 * (self_ctx + cross_ctx)
            else:
                context = self_ctx

            context = self.context_norm(context)
            gate = torch.sigmoid(self.gate(torch.cat([base, context], dim=-1)))
            fused.append(base + self.residual_weight * gate * context)

        return fused


class build_transformer(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, factory):
        super(build_transformer, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH_T
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.token_dim = 768
        self.trans_type = cfg.MODEL.TRANSFORMER_TYPE
        if 't2t' in cfg.MODEL.TRANSFORMER_TYPE:
            self.token_dim = 512
        if 'edge' in cfg.MODEL.TRANSFORMER_TYPE or cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224':
            self.token_dim = 384
        if '14' in cfg.MODEL.TRANSFORMER_TYPE:
            self.token_dim = 384
        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        num_classes=num_classes,
                                                        camera=camera_num, view=0,
                                                        stride_size=cfg.MODEL.STRIDE_SIZE,
                                                        drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate=cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

    def forward(self, x, cam_label, view_label=None):
        cash_x, attn = self.base(x, camera_id=cam_label, view_id=view_label)
        return cash_x, attn

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location='cpu')
        model_dict = self.state_dict()
        loaded, skipped = [], []
        for k, v in param_dict.items():
            clean_key = k.replace('module.', '')
            if clean_key not in model_dict:
                skipped.append(clean_key)
                continue
            try:
                model_dict[clean_key].copy_(v)
                loaded.append(clean_key)
            except Exception as e:
                skipped.append(clean_key)
                print('WARNING: skip key {}: checkpoint {} vs model {} ({})'.format(
                    clean_key, v.shape, model_dict[clean_key].shape, e))
        print('Loading pretrained model from {}'.format(trained_path))
        print('  Loaded {}/{} keys'.format(len(loaded), len(model_dict)))
        if skipped:
            print('  Skipped keys: {}'.format(skipped))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        model_dict = self.state_dict()
        for k, v in param_dict.items():
            if k in model_dict:
                try:
                    model_dict[k].copy_(v)
                except Exception as e:
                    print('WARNING: skip key {}: {} ({})'.format(k, e, v.shape))
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class HTLReID(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, factory):
        super(HTLReID, self).__init__()
        # Three Modalities share the same backbone
        self.BACKBONE = build_transformer(num_classes, cfg, camera_num, factory)
        self.feat_h = int(cfg.INPUT.SIZE_TRAIN[0] // cfg.MODEL.STRIDE_SIZE[0])
        self.feat_w = int(cfg.INPUT.SIZE_TRAIN[1] // cfg.MODEL.STRIDE_SIZE[1])
        self.num_patches = self.feat_h * self.feat_w
        self.HS_FACSS = HSFACSS(dim=self.BACKBONE.token_dim, cfg=cfg)
        self.FREQ_INDEX = Frequency_based_Token_Selection(keep=cfg.MODEL.FREQUENCY_KEEP,
                                                          stride=cfg.MODEL.STRIDE_SIZE[0],
                                                          quality_aware=cfg.MODEL.FREQUENCY_QUALITY_AWARE)
        self.use_agf = cfg.MODEL.AGF
        self.use_ocfr = bool(cfg.MODEL.OCFR)
        self.use_quality = bool(cfg.MODEL.QUALITY_AWARE)
        self.use_adapter = bool(cfg.MODEL.MODALITY_ADAPTER)
        self.use_part = bool(cfg.MODEL.PART_BRANCH)
        self.part_num = int(cfg.MODEL.PART_NUM)
        self.part_pool = cfg.MODEL.PART_POOL.lower()
        if self.part_pool not in ('stripe', 'semantic'):
            raise ValueError("MODEL.PART_POOL must be 'stripe' or 'semantic'")
        self.modality_drop_prob = float(cfg.INPUT.MODALITY_DROP_PROB)
        self.align_loss_weight = float(cfg.MODEL.ALIGN_LOSS_WEIGHT)
        self.token_consistency_weight = float(cfg.MODEL.TOKEN_CONSISTENCY_WEIGHT)
        self.gate_balance_weight = float(cfg.MODEL.GATE_BALANCE_WEIGHT)
        self.quality_perturb_loss_weight = float(cfg.MODEL.QUALITY_PERTURB_LOSS_WEIGHT)
        self.quality_min_score = float(cfg.MODEL.QUALITY_MIN_SCORE)
        self.test_part_feat = cfg.TEST.PART_FEAT.lower()
        self.test_part_feat_weight = float(cfg.TEST.PART_FEAT_WEIGHT)
        if self.test_part_feat not in ('off', 'concat', 'only'):
            raise ValueError("TEST.PART_FEAT must be 'off', 'concat', or 'only'")

        self.selected_patch_blend_weight = float(cfg.MODEL.SELECTED_PATCH_BLEND_WEIGHT)
        self.selected_patch_context = cfg.MODEL.SELECTED_PATCH_CONTEXT.lower()
        if self.selected_patch_context not in ('mean', 'attn_gate'):
            raise ValueError("MODEL.SELECTED_PATCH_CONTEXT must be 'mean' or 'attn_gate'")
        self.selected_patch_attn_scale = float(cfg.MODEL.SELECTED_PATCH_ATTN_SCALE)
        self.use_selected_aggregation = bool(cfg.MODEL.SELECTED_AGGREGATION)
        self.agf_residual_weight = float(cfg.MODEL.AGF_RESIDUAL_WEIGHT)
        self.agf_fusion_mode = cfg.MODEL.AGF_FUSION_MODE.lower()
        if self.agf_fusion_mode not in ('residual', 'agreement'):
            raise ValueError("MODEL.AGF_FUSION_MODE must be 'residual' or 'agreement'")
        self.agf_agree_min = float(cfg.MODEL.AGF_AGREE_MIN)
        self.agf_agree_temp = float(cfg.MODEL.AGF_AGREE_TEMP)
        self.agf_norm_cap = float(cfg.MODEL.AGF_NORM_CAP)
        if self.selected_patch_context == 'attn_gate':
            self.SELECTED_CONTEXT_NORM = nn.LayerNorm(self.BACKBONE.token_dim)
            self.SELECTED_CONTEXT_GATE = nn.Linear(2 * self.BACKBONE.token_dim,
                                                   self.BACKBONE.token_dim)
            nn.init.zeros_(self.SELECTED_CONTEXT_GATE.weight)
            nn.init.constant_(self.SELECTED_CONTEXT_GATE.bias,
                              float(cfg.MODEL.SELECTED_PATCH_GATE_INIT))
        if self.use_selected_aggregation:
            self.SELECTED_AGGREGATOR = SelectedTokenAggregator(
                dim=self.BACKBONE.token_dim,
                num_heads=cfg.MODEL.SELECTED_AGG_NUM_HEADS,
                gate_init_bias=cfg.MODEL.SELECTED_AGG_GATE_INIT_BIAS,
                residual_weight=cfg.MODEL.SELECTED_AGG_RESIDUAL_WEIGHT,
            )
        if self.use_agf:
            self.AGF = AGF(dim=self.BACKBONE.token_dim,
                           num_heads=cfg.MODEL.AGF_NUM_HEADS,
                           gate_init_bias=cfg.MODEL.AGF_GATE_INIT_BIAS,
                           quality_scale=bool(cfg.MODEL.AGF_QUALITY_SCALE),
                           mode=cfg.MODEL.AGF_MODE,
                           tpm_steps=cfg.MODEL.AGF_TPM_STEPS)
        if self.use_adapter:
            self.MODALITY_ADAPTERS = nn.ModuleDict({
                'RGB': ModalityAdapter(self.BACKBONE.token_dim,
                                       cfg.MODEL.MODALITY_ADAPTER_DIM,
                                       cfg.MODEL.MODALITY_ADAPTER_SCALE),
                'NIR': ModalityAdapter(self.BACKBONE.token_dim,
                                       cfg.MODEL.MODALITY_ADAPTER_DIM,
                                       cfg.MODEL.MODALITY_ADAPTER_SCALE),
                'TIR': ModalityAdapter(self.BACKBONE.token_dim,
                                       cfg.MODEL.MODALITY_ADAPTER_DIM,
                                       cfg.MODEL.MODALITY_ADAPTER_SCALE),
            })
        if self.use_quality:
            self.QUALITY_HEAD = ModalityQualityHead(
                dim=self.BACKBONE.token_dim,
                hidden=cfg.MODEL.QUALITY_HIDDEN,
                prior=cfg.MODEL.QUALITY_PRIOR,
                min_score=cfg.MODEL.QUALITY_MIN_SCORE,
            )
        if self.use_ocfr:
            self.memory_cls = OCFR(dim=self.BACKBONE.token_dim, num_class=num_classes, momentum=0.8)

        # The output learning params of fused features
        self.FUSE_HEAD = nn.Linear(3 * self.BACKBONE.token_dim, num_classes, bias=False)
        self.FUSE_BN = nn.BatchNorm1d(3 * self.BACKBONE.token_dim)
        self.FUSE_HEAD.apply(weights_init_classifier)

        # The output learning params of RGB/NIR/TIR cls tokens
        self.BACKBONE_HEAD = nn.Linear(self.BACKBONE.token_dim, num_classes, bias=False)
        self.BACKBONE_BN = nn.BatchNorm1d(self.BACKBONE.token_dim)
        self.BACKBONE_HEAD.apply(weights_init_classifier)
        # Here, you can choose to use different head for different modalities
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # self.BACKBONE_HEAD_2 = nn.Linear(self.BACKBONE.token_dim, num_classes, bias=False)
        # self.BACKBONE_BN_2 = nn.BatchNorm1d(self.BACKBONE.token_dim)
        # self.BACKBONE_HEAD_2.apply(weights_init_classifier)
        # self.BACKBONE_HEAD_3 = nn.Linear(self.BACKBONE.token_dim, num_classes, bias=False)
        # self.BACKBONE_BN_3 = nn.BatchNorm1d(self.BACKBONE.token_dim)
        # self.BACKBONE_HEAD_3.apply(weights_init_classifier)
        # If you use above head, you need to change the forward function to return the scores of different modalities
        # RGB_cls_score = self.BACKBONE_HEAD(self.BACKBONE_BN(RGB_cls4tri))
        # NIR_cls_score = self.BACKBONE_HEAD_2(self.BACKBONE_BN_2(NIR_cls4tri))
        # TIR_cls_score = self.BACKBONE_HEAD_3(self.BACKBONE_BN_3(TIR_cls4tri))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # In fact, you can choose the AL setting like TOP-ReID, here is the head for AL setting.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.AL = cfg.MODEL.AL
        if self.AL:
            self.AL_HEAD = nn.Linear(3 * self.BACKBONE.token_dim, num_classes, bias=False)
            self.AL_BN = nn.BatchNorm1d(3 * self.BACKBONE.token_dim)
            self.AL_HEAD.apply(weights_init_classifier)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.use_part:
            if self.part_pool == 'semantic':
                self.PART_QUERIES = nn.Parameter(
                    torch.empty(self.part_num, self.BACKBONE.token_dim))
                nn.init.normal_(self.PART_QUERIES, std=0.02)
            part_dim = 3 * self.part_num * self.BACKBONE.token_dim
            self.PART_BN = nn.BatchNorm1d(part_dim)
            self.PART_HEAD = nn.Linear(part_dim, num_classes, bias=False)
            self.PART_HEAD.apply(weights_init_classifier)

    @staticmethod
    def _prepare_patch_mask(patches, mask=None):
        if mask is None:
            return None
        valid = mask.to(device=patches.device, dtype=torch.bool).clone()
        empty = ~valid.any(dim=1)
        if empty.any():
            valid[empty] = True
        return valid

    def _pool_selected_patches(self, feat, mask=None):
        patches = feat[:, 1:, :]
        valid = self._prepare_patch_mask(patches, mask)
        if valid is None:
            return patches.mean(dim=1)
        weight = valid.to(dtype=patches.dtype)
        denom = weight.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (patches * weight.unsqueeze(-1)).sum(dim=1) / denom

    def _attend_selected_patches(self, cls_token, feat, mask=None):
        patches = feat[:, 1:, :]
        valid = self._prepare_patch_mask(patches, mask)
        query = F.normalize(cls_token, dim=-1).unsqueeze(1)
        key = F.normalize(patches, dim=-1)
        logits = (query * key).sum(dim=-1) * self.selected_patch_attn_scale
        if valid is not None:
            logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        weights = F.softmax(logits, dim=1)
        return (patches * weights.unsqueeze(-1)).sum(dim=1)

    def _cls_with_selected_context(self, feat, mask=None):
        cls_token = feat[:, 0, :]
        if self.selected_patch_blend_weight <= 0:
            return cls_token
        patch_context = self._pool_selected_patches(feat, mask)
        if self.selected_patch_context == 'attn_gate':
            attn_context = self._attend_selected_patches(cls_token, feat, mask)
            patch_context = 0.5 * (patch_context + attn_context)
            patch_context = self.SELECTED_CONTEXT_NORM(patch_context)
            gate = torch.sigmoid(
                self.SELECTED_CONTEXT_GATE(torch.cat([cls_token, patch_context], dim=-1))
            )
            patch_context = gate * patch_context
        return cls_token + self.selected_patch_blend_weight * patch_context

    def _concat_cls(self, rgb_feat, nir_feat, tir_feat, quality_scores=None, masks=None):
        masks = masks or (None, None, None)
        has_tir = len(masks) != 2 and tir_feat is not None
        if len(masks) == 2:
            masks = (masks[0], masks[1], None)
        cls_list = [
            self._cls_with_selected_context(rgb_feat, masks[0]),
            self._cls_with_selected_context(nir_feat, masks[1]),
            self._cls_with_selected_context(tir_feat, masks[2]),
        ]
        if self.use_selected_aggregation:
            cls_list = self.SELECTED_AGGREGATOR(
                (rgb_feat, nir_feat, tir_feat),
                masks=masks,
                active=(True, True, has_tir),
                base_cls=cls_list,
            )
        if quality_scores is not None:
            cls_list = [
                cls_list[i] * quality_scores[:, i:i + 1]
                for i in range(3)
            ]
        return torch.cat(cls_list, dim=-1)

    def _agf_cls(self, rgb_feat, nir_feat, tir_feat, quality_scores=None, masks=None):
        base_cls = self._concat_cls(rgb_feat, nir_feat, tir_feat,
                                    quality_scores, masks=masks)
        agf_cls = self.AGF(rgb_feat, nir_feat, tir_feat,
                           quality_scores=quality_scores)
        weight = max(0.0, min(1.0, self.agf_residual_weight))
        if self.agf_fusion_mode == 'residual':
            return base_cls + weight * (agf_cls - base_cls)

        base_nodes = base_cls.chunk(3, dim=-1)
        agf_nodes = agf_cls.chunk(3, dim=-1)
        fused_nodes = []
        for idx, (base_node, agf_node) in enumerate(zip(base_nodes, agf_nodes)):
            delta = agf_node - base_node
            if self.agf_norm_cap > 0:
                base_norm = base_node.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                delta_norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                max_delta = base_norm * self.agf_norm_cap
                delta = delta * (max_delta / delta_norm).clamp(max=1.0)

            agree = F.cosine_similarity(base_node, agf_node, dim=-1, eps=1e-6)
            gate = torch.sigmoid(
                (agree - self.agf_agree_min) * self.agf_agree_temp
            ).unsqueeze(-1)
            fused_nodes.append(base_node + weight * gate * delta)
        return torch.cat(fused_nodes, dim=-1)

    def _apply_modality_dropout(self, rgb, nir, tir=None, return_keep=False):
        if (not self.training) or self.modality_drop_prob <= 0:
            if not return_keep:
                return rgb, nir, tir
            modalities = 2 if tir is None else 3
            keep = torch.ones(rgb.size(0), modalities, device=rgb.device, dtype=torch.bool)
            return rgb, nir, tir, keep
        modalities = 2 if tir is None else 3
        keep = torch.rand(rgb.size(0), modalities, device=rgb.device) > self.modality_drop_prob
        empty = ~keep.any(dim=1)
        if empty.any():
            keep[empty, 1 if modalities > 1 else 0] = True
        rgb = rgb * keep[:, 0].view(-1, 1, 1, 1).to(rgb.dtype)
        nir = nir * keep[:, 1].view(-1, 1, 1, 1).to(nir.dtype)
        if tir is not None:
            tir = tir * keep[:, 2].view(-1, 1, 1, 1).to(tir.dtype)
        if return_keep:
            return rgb, nir, tir, keep
        return rgb, nir, tir

    def _adapt_features(self, rgb_feat, nir_feat, tir_feat=None):
        if not self.use_adapter:
            return rgb_feat, nir_feat, tir_feat
        rgb_feat = self.MODALITY_ADAPTERS['RGB'](rgb_feat)
        nir_feat = self.MODALITY_ADAPTERS['NIR'](nir_feat)
        if tir_feat is not None:
            tir_feat = self.MODALITY_ADAPTERS['TIR'](tir_feat)
        return rgb_feat, nir_feat, tir_feat

    def _stripe_part_feature(self, rgb_feat, nir_feat, tir_feat, quality_scores=None):
        def modal_parts(feat, quality=None):
            B, _, C = feat.shape
            patches = feat[:, 1:, :].view(B, self.feat_h, self.feat_w, C)
            pooled = []
            for chunk in torch.chunk(patches, self.part_num, dim=1):
                pooled.append(chunk.mean(dim=(1, 2)))
            out = torch.cat(pooled, dim=-1)
            if quality is not None:
                out = out * quality.view(B, 1)
            return out

        qualities = [None, None, None]
        if quality_scores is not None:
            qualities = [quality_scores[:, i] for i in range(3)]
        return torch.cat([
            modal_parts(rgb_feat, qualities[0]),
            modal_parts(nir_feat, qualities[1]),
            modal_parts(tir_feat, qualities[2]),
        ], dim=-1)

    def _semantic_part_feature(self, rgb_feat, nir_feat, tir_feat, quality_scores=None, masks=None):
        def modal_parts(feat, mask=None, quality=None):
            B, _, C = feat.shape
            patches = feat[:, 1:, :]
            queries = F.normalize(self.PART_QUERIES, dim=-1)
            patch_norm = F.normalize(patches, dim=-1)
            logits = torch.matmul(patch_norm, queries.t())
            if mask is not None:
                valid = mask.bool()
                logits = logits.masked_fill(~valid.unsqueeze(-1), torch.finfo(logits.dtype).min)
                empty = ~valid.any(dim=1)
                if empty.any():
                    logits[empty] = torch.matmul(
                        patch_norm[empty], queries.t())
            attn = torch.softmax(logits, dim=1)
            pooled = torch.einsum('bnp,bnd->bpd', attn, patches).reshape(B, self.part_num * C)
            if quality is not None:
                pooled = pooled * quality.view(B, 1)
            return pooled

        qualities = [None, None, None]
        if quality_scores is not None:
            qualities = [quality_scores[:, i] for i in range(3)]
        masks = masks or (None, None, None)
        if len(masks) == 2:
            masks = (masks[0], masks[1], None)
        return torch.cat([
            modal_parts(rgb_feat, masks[0], qualities[0]),
            modal_parts(nir_feat, masks[1], qualities[1]),
            modal_parts(tir_feat, masks[2], qualities[2]),
        ], dim=-1)

    def _part_feature(self, rgb_feat, nir_feat, tir_feat, quality_scores=None, masks=None):
        if self.part_pool == 'semantic':
            return self._semantic_part_feature(rgb_feat, nir_feat, tir_feat,
                                               quality_scores=quality_scores, masks=masks)
        return self._stripe_part_feature(rgb_feat, nir_feat, tir_feat,
                                         quality_scores=quality_scores)

    def _test_descriptor(self, cls4t, rgb_feat, nir_feat, tir_feat, quality_scores=None, masks=None):
        if self.test_part_feat == 'off' or not self.use_part:
            return cls4t

        part_feat = self._part_feature(rgb_feat, nir_feat, tir_feat, quality_scores, masks)
        part_feat = F.normalize(part_feat, dim=-1) * self.test_part_feat_weight
        if self.test_part_feat == 'only':
            return part_feat

        cls4t = F.normalize(cls4t, dim=-1)
        return torch.cat([cls4t, part_feat], dim=-1)

    def _auxiliary_losses(self, rgb_feat, nir_feat, tir_feat, masks, quality_scores, has_tir=True):
        loss = torch.zeros((), device=rgb_feat.device)
        cls_list = [rgb_feat[:, 0, :], nir_feat[:, 0, :], tir_feat[:, 0, :]]
        if quality_scores is None:
            quality_scores = torch.ones(rgb_feat.size(0), 3, device=rgb_feat.device, dtype=rgb_feat.dtype)
            if not has_tir:
                quality_scores[:, 2] = 0.0

        valid_pairs = [(0, 1)]
        if has_tir:
            valid_pairs += [(0, 2), (1, 2)]

        if self.align_loss_weight > 0:
            align = torch.zeros_like(loss)
            norm_cls = [F.normalize(c, dim=-1) for c in cls_list]
            for i, j in valid_pairs:
                weight = quality_scores[:, i] * quality_scores[:, j]
                align = align + (weight * (1.0 - (norm_cls[i] * norm_cls[j]).sum(dim=-1))).mean()
            loss = loss + self.align_loss_weight * align / max(len(valid_pairs), 1)

        if self.token_consistency_weight > 0 and masks is not None:
            token_loss = torch.zeros_like(loss)
            for i, j in valid_pairs:
                mi = masks[i].float()
                mj = masks[j].float()
                inter = (mi * mj).sum(dim=1)
                union = ((mi + mj) > 0).float().sum(dim=1).clamp_min(1.0)
                weight = quality_scores[:, i] * quality_scores[:, j]
                token_loss = token_loss + (weight * (1.0 - inter / union)).mean()
            loss = loss + self.token_consistency_weight * token_loss / max(len(valid_pairs), 1)

        if self.gate_balance_weight > 0:
            active = torch.tensor([1.0, 1.0, 1.0 if has_tir else 0.0],
                                  device=rgb_feat.device, dtype=quality_scores.dtype).view(1, 3)
            q = quality_scores * active
            q_norm = q / (q.sum(dim=1, keepdim=True) + 1e-6)
            target = active / active.sum(dim=1, keepdim=True).clamp_min(1.0)
            loss = loss + self.gate_balance_weight * ((q_norm - target) ** 2).sum(dim=1).mean()

        return loss

    def _quality_dropout_loss(self, quality_scores, keep_mask):
        if quality_scores is None or keep_mask is None or self.quality_perturb_loss_weight <= 0:
            device = keep_mask.device if keep_mask is not None else next(self.parameters()).device
            return torch.zeros((), device=device)
        width = keep_mask.size(1)
        dropped = (~keep_mask).to(dtype=quality_scores.dtype)
        if dropped.sum().item() <= 0:
            return torch.zeros((), device=quality_scores.device)
        quality = quality_scores[:, :width]
        target = torch.full_like(quality, self.quality_min_score)
        loss = ((quality - target) ** 2 * dropped).sum() / dropped.sum().clamp_min(1.0)
        return self.quality_perturb_loss_weight * loss

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location='cpu')
        model_dict = self.state_dict()
        loaded, skipped = [], []
        for k, v in param_dict.items():
            clean_key = k.replace('module.', '')
            if clean_key not in model_dict:
                skipped.append(clean_key)
                continue
            try:
                model_dict[clean_key].copy_(v)
                loaded.append(clean_key)
            except Exception as e:
                skipped.append(clean_key)
                print('WARNING: skip key {}: checkpoint {} vs model {} ({})'.format(
                    clean_key, v.shape, model_dict[clean_key].shape, e))
        missing_keys = [k for k in model_dict if k not in loaded]
        print('Loading pretrained model from {}'.format(trained_path))
        print('  Loaded {}/{} keys'.format(len(loaded), len(model_dict)))
        if skipped:
            print('  Skipped keys (in checkpoint but not loaded): {}'.format(skipped))
        if missing_keys:
            print('  Missing keys (randomly initialized): {}'.format(missing_keys))

    def forward(self, x, cam_label=None, label=None, view_label=None, img_path=None, mode=1,
                writer=None, epoch=None):
        if self.training:
            RGB = x['RGB']
            NIR = x['NI']
            TIR = x['TI']
            RGB, NIR, TIR, keep_mask = self._apply_modality_dropout(
                RGB, NIR, TIR, return_keep=True)
            RGB_feat, RGB_attn = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label)
            NIR_feat, NIR_attn = self.BACKBONE(NIR, cam_label=cam_label, view_label=view_label)
            TIR_feat, TIR_attn = self.BACKBONE(TIR, cam_label=cam_label, view_label=view_label)
            RGB_feat, NIR_feat, TIR_feat = self._adapt_features(RGB_feat, NIR_feat, TIR_feat)

            RGB_cls4tri = RGB_feat[:, 0, :]
            NIR_cls4tri = NIR_feat[:, 0, :]
            TIR_cls4tri = TIR_feat[:, 0, :]
            quality_scores = self.QUALITY_HEAD(RGB_cls4tri, NIR_cls4tri, TIR_cls4tri) \
                if self.use_quality else None
            mask_fre = self.FREQ_INDEX(x=RGB, y=NIR, z=TIR, img_path=img_path, mode=mode, writer=writer,
                                       step=epoch, quality_scores=quality_scores)
            if self.AL:
                ori = torch.cat([RGB_cls4tri, NIR_cls4tri, TIR_cls4tri], dim=-1)
                ori_score = self.AL_HEAD(self.AL_BN(ori))
            else:
                RGB_cls_score = self.BACKBONE_HEAD(self.BACKBONE_BN(RGB_cls4tri))
                NIR_cls_score = self.BACKBONE_HEAD(self.BACKBONE_BN(NIR_cls4tri))
                TIR_cls_score = self.BACKBONE_HEAD(self.BACKBONE_BN(TIR_cls4tri))

            RGB_feat_s, NIR_feat_s, TIR_feat_s, mask = self.HS_FACSS(RGB_feat=RGB_feat,
                                                                     RGB_attn=RGB_attn,
                                                                     NIR_feat=NIR_feat,
                                                                     NIR_attn=NIR_attn,
                                                                     TIR_feat=TIR_feat,
                                                                     TIR_attn=TIR_attn,
                                                                     img_path=img_path,
                                                                     epoch=epoch, writer=writer,
                                                                     mask_fre=mask_fre,
                                                                     quality_scores=quality_scores)

            if self.use_agf:
                cls4t = self._agf_cls(RGB_feat_s, NIR_feat_s, TIR_feat_s,
                                      quality_scores, masks=mask)
            else:
                cls4t = self._concat_cls(RGB_feat_s, NIR_feat_s, TIR_feat_s,
                                         quality_scores, masks=mask)
            if self.use_ocfr:
                RGB_cls = RGB_feat_s[:, 0, :]
                NIR_cls = NIR_feat_s[:, 0, :]
                TIR_cls = TIR_feat_s[:, 0, :]
                loss_aux = self.memory_cls(RGB_cls, NIR_cls, TIR_cls, label, epoch=epoch)
            else:
                loss_aux = torch.zeros((), device=RGB_feat.device)
            loss_aux = loss_aux + self._auxiliary_losses(
                RGB_feat_s, NIR_feat_s, TIR_feat_s, mask, quality_scores, has_tir=True)
            loss_aux = loss_aux + self._quality_dropout_loss(quality_scores, keep_mask)
            score = self.FUSE_HEAD(self.FUSE_BN(cls4t))
            if self.use_part:
                part_feat = self._part_feature(RGB_feat_s, NIR_feat_s, TIR_feat_s, quality_scores, masks=mask)
                part_score = self.PART_HEAD(self.PART_BN(part_feat))
            if self.AL:
                if self.use_part:
                    return score, cls4t, ori_score, ori, part_score, part_feat, loss_aux
                return score, cls4t, ori_score, ori, loss_aux
            else:
                if self.use_part:
                    return score, cls4t, RGB_cls_score, RGB_cls4tri, NIR_cls_score, NIR_cls4tri, TIR_cls_score, TIR_cls4tri, part_score, part_feat, loss_aux
                return score, cls4t, RGB_cls_score, RGB_cls4tri, NIR_cls_score, NIR_cls4tri, TIR_cls_score, TIR_cls4tri, loss_aux
        else:
            RGB = x['RGB']
            NIR = x['NI']
            TIR = x['TI']
            RGB_feat, RGB_attn = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label)
            NIR_feat, NIR_attn = self.BACKBONE(NIR, cam_label=cam_label, view_label=view_label)
            TIR_feat, TIR_attn = self.BACKBONE(TIR, cam_label=cam_label, view_label=view_label)
            RGB_feat, NIR_feat, TIR_feat = self._adapt_features(RGB_feat, NIR_feat, TIR_feat)
            quality_scores = self.QUALITY_HEAD(RGB_feat[:, 0, :], NIR_feat[:, 0, :], TIR_feat[:, 0, :]) \
                if self.use_quality else None
            mask_fre = self.FREQ_INDEX(x=RGB, y=NIR, z=TIR, img_path=img_path, mode=mode, writer=writer,
                                       step=epoch, quality_scores=quality_scores)

            RGB_feat_s, NIR_feat_s, TIR_feat_s, mask = self.HS_FACSS(RGB_feat=RGB_feat,
                                                                     RGB_attn=RGB_attn,
                                                                     NIR_feat=NIR_feat,
                                                                     NIR_attn=NIR_attn,
                                                                     TIR_feat=TIR_feat,
                                                                     TIR_attn=TIR_attn,
                                                                     img_path=img_path,
                                                                     epoch=epoch, writer=writer,
                                                                     mask_fre=mask_fre,
                                                                     quality_scores=quality_scores)

            if self.use_agf:
                cls4t = self._agf_cls(RGB_feat_s, NIR_feat_s, TIR_feat_s,
                                      quality_scores, masks=mask)
            else:
                cls4t = self._concat_cls(RGB_feat_s, NIR_feat_s, TIR_feat_s,
                                         quality_scores, masks=mask)
            return self._test_descriptor(cls4t, RGB_feat_s, NIR_feat_s, TIR_feat_s,
                                         quality_scores, masks=mask)

    def forward_two_modalities(self, x, cam_label=None, label=None, view_label=None, cross_type=None, img_path=None,
                               mode=1,
                               writer=None, epoch=None):
        # This forward function is used for the two modalities datasets like RGBN300
        if self.training:
            RGB = x['RGB']
            NIR = x['NI']
            RGB, NIR, _, keep_mask = self._apply_modality_dropout(
                RGB, NIR, None, return_keep=True)
            RGB_feat, RGB_attn = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label)
            NIR_feat, NIR_attn = self.BACKBONE(NIR, cam_label=cam_label, view_label=view_label)
            RGB_feat, NIR_feat, _ = self._adapt_features(RGB_feat, NIR_feat, None)

            RGB_cls4tri = RGB_feat[:, 0, :]
            NIR_cls4tri = NIR_feat[:, 0, :]
            quality_scores = self.QUALITY_HEAD(RGB_cls4tri, NIR_cls4tri, None) \
                if self.use_quality else None
            mask_fre = self.FREQ_INDEX(x=RGB, y=NIR, z=None, img_path=img_path, mode=mode, writer=writer,
                                       step=epoch, quality_scores=quality_scores)
            # Here, you need to change the head for the AL setting to 2*token_dim
            if self.AL:
                ori = torch.cat([RGB_cls4tri, NIR_cls4tri], dim=-1)
                ori_score = self.AL_HEAD(self.AL_BN(ori))
            else:
                RGB_cls_score = self.BACKBONE_HEAD(self.BACKBONE_BN(RGB_cls4tri))
                NIR_cls_score = self.BACKBONE_HEAD(self.BACKBONE_BN(NIR_cls4tri))

            RGB_feat_s, NIR_feat_s, mask = self.HS_FACSS(RGB_feat=RGB_feat,
                                                         RGB_attn=RGB_attn,
                                                         NIR_feat=NIR_feat,
                                                         NIR_attn=NIR_attn,
                                                         TIR_feat=None,
                                                         TIR_attn=None,
                                                         img_path=img_path,
                                                         epoch=epoch, writer=writer,
                                                         mask_fre=mask_fre,
                                                         quality_scores=quality_scores)

            TIR_feat_s = torch.zeros_like(RGB_feat_s)
            if self.use_agf:
                cls4t = self._agf_cls(RGB_feat_s, NIR_feat_s, TIR_feat_s,
                                      quality_scores, masks=mask)
            else:
                cls4t = self._concat_cls(RGB_feat_s, NIR_feat_s, TIR_feat_s,
                                         quality_scores, masks=mask)
            if self.use_ocfr:
                RGB_cls = RGB_feat_s[:, 0, :]
                NIR_cls = NIR_feat_s[:, 0, :]
                TIR_cls = TIR_feat_s[:, 0, :]
                loss_aux = self.memory_cls(RGB_cls, NIR_cls, TIR_cls, label, epoch=epoch)
            else:
                loss_aux = torch.zeros((), device=RGB_feat.device)
            loss_aux = loss_aux + self._auxiliary_losses(
                RGB_feat_s, NIR_feat_s, TIR_feat_s, mask, quality_scores, has_tir=False)
            loss_aux = loss_aux + self._quality_dropout_loss(quality_scores, keep_mask)
            score = self.FUSE_HEAD(self.FUSE_BN(cls4t))
            if self.use_part:
                part_feat = self._part_feature(RGB_feat_s, NIR_feat_s, TIR_feat_s, quality_scores, masks=mask)
                part_score = self.PART_HEAD(self.PART_BN(part_feat))
            if self.AL:
                if self.use_part:
                    return score, cls4t, ori_score, ori, part_score, part_feat, loss_aux
                return score, cls4t, ori_score, ori, loss_aux
            else:
                if self.use_part:
                    return score, cls4t, RGB_cls_score, RGB_cls4tri, NIR_cls_score, NIR_cls4tri, part_score, part_feat, loss_aux
                return score, cls4t, RGB_cls_score, RGB_cls4tri, NIR_cls_score, NIR_cls4tri, loss_aux

        else:
            RGB = x['RGB']
            NIR = x['NI']
            RGB_feat, RGB_attn = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label)
            NIR_feat, NIR_attn = self.BACKBONE(NIR, cam_label=cam_label, view_label=view_label)
            RGB_feat, NIR_feat, _ = self._adapt_features(RGB_feat, NIR_feat, None)
            quality_scores = self.QUALITY_HEAD(RGB_feat[:, 0, :], NIR_feat[:, 0, :], None) \
                if self.use_quality else None
            mask_fre = self.FREQ_INDEX(x=RGB, y=NIR, z=None, img_path=img_path, mode=mode, writer=writer,
                                       step=epoch, quality_scores=quality_scores)

            RGB_feat_s, NIR_feat_s, mask = self.HS_FACSS(RGB_feat=RGB_feat,
                                                         RGB_attn=RGB_attn,
                                                         NIR_feat=NIR_feat,
                                                         NIR_attn=NIR_attn,
                                                         TIR_feat=None,
                                                         TIR_attn=None,
                                                         img_path=img_path,
                                                         epoch=epoch, writer=writer,
                                                         mask_fre=mask_fre,
                                                         quality_scores=quality_scores)

            TIR_feat_s = torch.zeros_like(RGB_feat_s)
            if self.use_agf:
                cls4t = self._agf_cls(RGB_feat_s, NIR_feat_s, TIR_feat_s,
                                      quality_scores, masks=mask)
            else:
                cls4t = self._concat_cls(RGB_feat_s, NIR_feat_s, TIR_feat_s,
                                         quality_scores, masks=mask)
            return self._test_descriptor(cls4t, RGB_feat_s, NIR_feat_s, TIR_feat_s,
                                         quality_scores, masks=mask)


__factory_T_type = {
    'vit_base_patch16_224': vit_base_patch16_224,
    'deit_base_patch16_224': vit_base_patch16_224,
    'vit_small_patch16_224': vit_small_patch16_224,
    'deit_small_patch16_224': deit_small_patch16_224,
}


def make_model(cfg, num_class, camera_num):
    model = HTLReID(num_class, cfg, camera_num, __factory_T_type)
    print('===========Building HTL-ReID===========')
    return model
