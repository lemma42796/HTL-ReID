import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalRollout(nn.Module):
    """
    Hierarchical Token Selection (HS).

    Cumulative attention rollout with head-mean and 0.5*I residual mixing
    (Abnar & Zuidema, 2020). Captures top-k indices at multiple ViT depths
    and unions them per modality.
    """

    def __init__(self, layers=(4, 8, 12), k=16):
        super().__init__()
        self.layers = tuple(sorted(layers))
        self.k = k

    def forward(self, attn_list):
        """
        attn_list: list of L tensors, each [B, H, M, M] where M = N + 1.

        Returns:
          union_mask: [B, N] bool, True at positions in J_m^hier.
          s_final:    [B, N] cumulative attention score at the deepest
                      requested layer; reused as S_self in FACSS.
          per_depth_scores: dict {layer: [B, N]} for visualization/debug.
        """
        max_layer = max(self.layers)
        device = attn_list[0].device
        B, _, M, _ = attn_list[0].shape
        N = M - 1

        eye = torch.eye(M, device=device).unsqueeze(0)  # [1, M, M]
        # Only the class-token row is consumed below. Building every full
        # cumulative [M, M] product wastes O(M^3) work at each layer. For a
        # requested depth l, e0^T A_l ... A_1 can instead be evaluated as a
        # chain of batched vector-matrix products with O(M^2) work.
        transition = []
        for l in range(1, max_layer + 1):
            A = attn_list[l - 1].mean(dim=1)            # [B, M, M]
            transition.append(0.5 * A + 0.5 * eye)

        snapshots = {}
        for l in self.layers:
            cls_row = transition[l - 1][:, 0, :]
            for prev in range(l - 2, -1, -1):
                cls_row = torch.bmm(
                    cls_row.unsqueeze(1), transition[prev]
                ).squeeze(1)
            snapshots[l] = cls_row[:, 1:]               # [B, N]

        union_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        for l in self.layers:
            s_l = snapshots[l]
            k = min(self.k, N)
            topk_idx = s_l.topk(k, dim=1).indices       # [B, k]
            mask_l = torch.zeros_like(union_mask)
            mask_l.scatter_(1, topk_idx, True)
            union_mask = union_mask | mask_l

        s_final = snapshots[max_layer]
        return union_mask, s_final, snapshots


class HSFACSS(nn.Module):
    """
    Hierarchical Selection (HS) + Fusion-Aware Synergistic Selection (FACSS).

    Replaces SFTS. Per-modality independent selection: HS produces a
    candidate set, FACSS rescues fusion-aware top-K via intra-modal
    discriminability + cross-modal cosine consensus modulated by an
    environment-aware scalar alpha.
    """

    def __init__(self, dim, cfg):
        super().__init__()
        self.dim = dim
        self.use_hs = bool(cfg.MODEL.HS_ENABLED)
        self.use_facss = bool(cfg.MODEL.FACSS_ENABLED)
        if self.use_facss and not self.use_hs:
            raise ValueError("MODEL.FACSS_ENABLED requires MODEL.HS_ENABLED")
        self.hs = HierarchicalRollout(
            layers=tuple(cfg.MODEL.HS_LAYERS),
            k=cfg.MODEL.HS_K,
        )
        self.facss_k = cfg.MODEL.FACSS_K
        self.dynamic_k = bool(cfg.MODEL.FACSS_DYNAMIC_K)
        self.facss_min_k = cfg.MODEL.FACSS_MIN_K
        self.facss_max_k = cfg.MODEL.FACSS_MAX_K
        self.soft_residual_weight = float(cfg.MODEL.FACSS_SOFT_RESIDUAL_WEIGHT)
        self.cross_pool = cfg.MODEL.FACSS_CROSS_POOL
        self.cross_topk = cfg.MODEL.FACSS_CROSS_TOPK
        self.cross_lse_tau = cfg.MODEL.FACSS_CROSS_LSE_TAU
        self.alpha_grain = cfg.MODEL.FACSS_ALPHA_GRANULARITY
        self.norm_mode = cfg.MODEL.FACSS_NORM
        self.use_ste = bool(cfg.MODEL.FACSS_STE)
        self.ste_tau = float(cfg.MODEL.FACSS_STE_TAU)
        self.modality_union = bool(cfg.MODEL.FACSS_MODALITY_UNION)
        self.union_promote = bool(cfg.MODEL.FACSS_UNION_PROMOTE)

        if self.use_facss:
            hidden = cfg.MODEL.FACSS_ALPHA_HIDDEN
            in_dim = 3 * dim if self.alpha_grain == 'sample' else 4 * dim
            self.alpha_mlp = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
            nn.init.zeros_(self.alpha_mlp[-1].bias)     # initial alpha ~= 0.5

            k_hidden = cfg.MODEL.FACSS_K_HIDDEN
            self.k_mlp = nn.Sequential(
                nn.LayerNorm(4 * dim + 1),
                nn.Linear(4 * dim + 1, k_hidden),
                nn.GELU(),
                nn.Linear(k_hidden, 1),
            )
            nn.init.zeros_(self.k_mlp[-1].weight)
            nn.init.zeros_(self.k_mlp[-1].bias)

    @staticmethod
    def _normalize(s, mode, mask=None, eps=1e-6):
        """
        Normalize s [B, N] per-sample, optionally restricted to candidate
        positions via bool mask [B, N] (non-candidate values are excluded
        from min/max/mean/std and zeroed in the output).
        """
        if mask is not None:
            very_neg = torch.finfo(s.dtype).min
            very_pos = torch.finfo(s.dtype).max
            s_for_max = torch.where(mask, s, torch.full_like(s, very_neg))
            s_for_min = torch.where(mask, s, torch.full_like(s, very_pos))
        else:
            s_for_max = s
            s_for_min = s

        if mode == 'minmax':
            lo = s_for_min.min(dim=1, keepdim=True).values
            hi = s_for_max.max(dim=1, keepdim=True).values
            out = (s - lo) / (hi - lo + eps)
        elif mode == 'robust':
            # quantile across candidate positions only
            if mask is not None:
                out = torch.zeros_like(s)
                for b in range(s.size(0)):
                    sb = s[b][mask[b]]
                    if sb.numel() == 0:
                        continue
                    lo = torch.quantile(sb, 0.05)
                    hi = torch.quantile(sb, 0.95)
                    sb_clipped = sb.clamp(lo, hi)
                    out[b][mask[b]] = (sb_clipped - lo) / (hi - lo + eps)
                return out
            lo = torch.quantile(s, 0.05, dim=1, keepdim=True)
            hi = torch.quantile(s, 0.95, dim=1, keepdim=True)
            s_clip = s.clamp(lo, hi)
            out = (s_clip - lo) / (hi - lo + eps)
        elif mode == 'zscore':
            if mask is not None:
                cnt = mask.sum(dim=1, keepdim=True).clamp(min=1).float()
                mu = (s * mask).sum(dim=1, keepdim=True) / cnt
                var = (((s - mu) * mask) ** 2).sum(dim=1, keepdim=True) / cnt
                sd = var.sqrt()
            else:
                mu = s.mean(dim=1, keepdim=True)
                sd = s.std(dim=1, keepdim=True)
            out = (s - mu) / (sd + eps)
        else:
            raise ValueError(mode)

        if mask is not None:
            out = out * mask.to(out.dtype)
        return out

    def _pool_cross(self, cos):
        if self.cross_pool == 'max':
            return cos.max(dim=-1).values
        if self.cross_pool == 'topk':
            k = min(self.cross_topk, cos.size(-1))
            return cos.topk(k, dim=-1).values.mean(dim=-1)
        if self.cross_pool == 'lse':
            tau = self.cross_lse_tau
            return (1.0 / tau) * torch.logsumexp(tau * cos, dim=-1)
        raise ValueError(self.cross_pool)

    def _cross_scores(self, full_patches, quality=None):
        """Compute each unordered modality pair once and reuse its transpose."""
        names = list(full_patches)
        normalized = {
            name: F.normalize(full_patches[name], dim=-1)
            for name in names
        }
        pooled = {name: [] for name in names}
        source_names = {name: [] for name in names}

        for left_idx, left in enumerate(names):
            for right in names[left_idx + 1:]:
                cos = torch.matmul(
                    normalized[left], normalized[right].transpose(-1, -2)
                ).relu_()
                pooled[left].append(self._pool_cross(cos))
                source_names[left].append(right)
                pooled[right].append(self._pool_cross(cos.transpose(-1, -2)))
                source_names[right].append(left)

        scores = {}
        for name in names:
            stacked = torch.stack(pooled[name], dim=1)             # [B, O, N]
            if quality is None:
                scores[name] = stacked.mean(dim=1)
                continue
            weights = torch.stack(
                [quality[source] for source in source_names[name]], dim=1
            ).unsqueeze(-1).clamp_min(0.0)                         # [B, O, 1]
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
            scores[name] = (stacked * weights).sum(dim=1)
        return scores

    def _alpha(self, cls_list, p_feats=None):
        """
        cls_list: list of [B, D] class tokens from each modality.
        p_feats:  [B, N, D] only used when granularity == 'token'.
        Returns alpha: [B, 1] (sample) or [B, N] (token).
        """
        ctx = torch.cat(cls_list, dim=-1)                          # [B, kD]
        if self.alpha_grain == 'sample':
            return torch.sigmoid(self.alpha_mlp(ctx))              # [B, 1]
        # token-level: broadcast ctx and concat per-token feature
        B, N, D = p_feats.shape
        ctx_exp = ctx.unsqueeze(1).expand(-1, N, -1)               # [B, N, kD]
        inp = torch.cat([p_feats, ctx_exp], dim=-1)                # [B, N, (k+1)D]
        return torch.sigmoid(self.alpha_mlp(inp).squeeze(-1))      # [B, N]

    def _predict_k(self, m_cls, cls_list, modality_quality, n_tokens):
        if not self.dynamic_k:
            k = min(self.facss_k, n_tokens)
            k_each = torch.full((m_cls.size(0),), k, dtype=torch.long, device=m_cls.device)
            ratio = torch.ones(m_cls.size(0), dtype=m_cls.dtype, device=m_cls.device)
            return k_each, ratio

        if modality_quality is None:
            modality_quality = torch.full((m_cls.size(0), 1), 0.5,
                                          dtype=m_cls.dtype, device=m_cls.device)
        elif modality_quality.dim() == 1:
            modality_quality = modality_quality.unsqueeze(-1)

        ctx = torch.cat(cls_list, dim=-1)
        ratio = torch.sigmoid(self.k_mlp(torch.cat([m_cls, ctx, modality_quality], dim=-1))).squeeze(-1)
        min_k = min(self.facss_min_k, n_tokens)
        max_k = min(max(self.facss_max_k, min_k), n_tokens)
        k_float = min_k + (max_k - min_k) * ratio
        k_each = k_float.round().long().clamp(min=min_k, max=max_k)
        return k_each, ratio

    def _select_one_modality(self, m_feat, m_attn, s_cross_full,
                             cls_list, mask_fre,
                             modality_quality=None):
        """
        Run HS + FACSS for one modality.
        m_feat: [B, M, D] full token sequence (cls + N patches).
        m_attn: list of L attention tensors.
        s_cross_full: cached cross-modal consensus score [B, N] or None.
        cls_list: [cls_rgb, cls_nir, cls_tir] from un-modified backbone.
        mask_fre: [B, N] bool or None.
        Returns: feat_out [B, M, D] (zero-padded outside selection),
                 sel_mask [B, N] bool of finally selected positions,
                 guide_score [B, N] continuous full-token importance.
        """
        cls_token = m_feat[:, :1, :]                               # [B, 1, D]
        patches = m_feat[:, 1:, :]                                 # [B, N, D]
        B, N, D = patches.shape

        if not self.use_hs:
            full_mask = torch.ones(B, N, dtype=torch.bool, device=patches.device)
            full_score = torch.ones(B, N, dtype=patches.dtype, device=patches.device)
            return m_feat, full_mask, full_score

        hs_mask, s_self_full, _ = self.hs(m_attn)                  # both per modality
        cand_mask = hs_mask
        if mask_fre is not None:
            cand_mask = cand_mask | mask_fre.bool()

        if not self.use_facss:
            gate = cand_mask.to(dtype=patches.dtype)
            feat_out = torch.cat(
                [cls_token, patches * gate.unsqueeze(-1)],
                dim=1,
            )
            guide_score = self._normalize(s_self_full, 'minmax')
            return feat_out, cand_mask, guide_score

        if s_cross_full is None:
            raise ValueError('FACSS requires a cached cross-modal score')

        s_self_norm = self._normalize(s_self_full, self.norm_mode, mask=cand_mask)
        s_cross_norm = self._normalize(s_cross_full, self.norm_mode, mask=cand_mask)
        guide_self = self._normalize(s_self_full, self.norm_mode)
        guide_cross = self._normalize(s_cross_full, self.norm_mode)

        if self.alpha_grain == 'sample':
            alpha = self._alpha(cls_list)                          # [B, 1]
            s_facss = s_self_norm + alpha * s_cross_norm
            guide_score = guide_self + alpha * guide_cross
        else:
            alpha = self._alpha(cls_list, p_feats=patches)         # [B, N]
            s_facss = s_self_norm + alpha * s_cross_norm
            guide_score = guide_self + alpha * guide_cross
        # FACR consumes a dense, calibrated score over every patch. Candidate
        # restriction remains exclusive to the legacy hard-selection path.
        guide_score = self._normalize(guide_score, 'minmax')

        # Restrict scoring to candidate positions: non-candidates get -inf
        very_neg = torch.finfo(s_facss.dtype).min
        s_facss = torch.where(cand_mask, s_facss, torch.full_like(s_facss, very_neg))

        k_each, k_ratio = self._predict_k(cls_token.squeeze(1), cls_list, modality_quality, N)
        max_select_k = min(
            self.facss_max_k if self.dynamic_k else self.facss_k,
            N,
        )
        topk_idx = s_facss.topk(max_select_k, dim=1).indices
        active_rank = torch.arange(max_select_k, device=patches.device).unsqueeze(0)
        active_rank = active_rank < k_each.unsqueeze(1)
        sel_mask = torch.zeros(B, N, dtype=torch.bool, device=patches.device)
        sel_mask.scatter_(1, topk_idx, active_rank)

        # Build the gating tensor applied to patches.
        # STE: forward = hard sel_mask; backward = softmax over candidate scores
        # (so alpha_mlp and the rollout-derived scores receive gradient).
        sel_mask_f = sel_mask.float()
        if self.use_ste and self.training:
            soft = F.softmax(s_facss / self.ste_tau, dim=1)        # [B, N]
            gate = sel_mask_f + soft - soft.detach()
        else:
            gate = sel_mask_f
        if self.soft_residual_weight > 0:
            if not (self.use_ste and self.training):
                soft = F.softmax(s_facss / self.ste_tau, dim=1)
            soft_scale = 0.5 + k_ratio.unsqueeze(-1)
            gate = gate + self.soft_residual_weight * soft_scale * soft * (1.0 - sel_mask_f)

        feat_out = torch.cat(
            [cls_token, patches * gate.unsqueeze(-1)],
            dim=1,
        )
        return feat_out, sel_mask, guide_score

    def forward(self, RGB_feat, RGB_attn, NIR_feat=None, NIR_attn=None,
                TIR_feat=None, TIR_attn=None, img_path=None,
                writer=None, epoch=None, mask_fre=None,
                quality_scores=None, return_scores=False):
        cls_rgb = RGB_feat[:, 0, :]
        patches_rgb = RGB_feat[:, 1:, :]

        modalities = [('RGB', RGB_feat, RGB_attn, patches_rgb, cls_rgb)]
        if NIR_feat is not None:
            modalities.append(('NIR', NIR_feat, NIR_attn, NIR_feat[:, 1:, :], NIR_feat[:, 0, :]))
        if TIR_feat is not None:
            modalities.append(('TIR', TIR_feat, TIR_attn, TIR_feat[:, 1:, :], TIR_feat[:, 0, :]))

        cls_list = [m[4] for m in modalities]
        # alpha_mlp is sized for 3 modalities; pad with a zero cls token when
        # fewer modalities are present (e.g. 2-modal datasets like RGBN300).
        # Zero context degrades alpha gracefully toward its bias-only response.
        while len(cls_list) < 3:
            cls_list.append(torch.zeros_like(cls_list[0]))
        full_patches = {m[0]: m[3] for m in modalities}
        quality = None
        if quality_scores is not None:
            quality = {
                'RGB': quality_scores[:, 0],
                'NIR': quality_scores[:, 1],
                'TIR': quality_scores[:, 2],
            }

        cross_scores = None
        if self.use_facss:
            cross_scores = self._cross_scores(full_patches, quality=quality)

        outs = {}
        masks = {}
        scores = {}
        for name, feat, attn, _, _ in modalities:
            modality_quality = None
            if quality is not None:
                modality_quality = quality[name]
            feat_out, sel_mask, guide_score = self._select_one_modality(
                feat, attn, None if cross_scores is None else cross_scores[name],
                cls_list, mask_fre,
                modality_quality,
            )
            outs[name] = feat_out
            masks[name] = sel_mask
            scores[name] = guide_score

        if self.use_facss and self.modality_union and len(masks) > 1:
            shared_mask = None
            for sel_mask in masks.values():
                shared_mask = sel_mask if shared_mask is None else (shared_mask | sel_mask)
            if self.union_promote:
                for name, feat, _, _, _ in modalities:
                    own_mask = masks[name]
                    extra_mask = (shared_mask & ~own_mask).unsqueeze(-1)
                    promoted = torch.where(extra_mask, feat[:, 1:, :], outs[name][:, 1:, :])
                    outs[name] = torch.cat([outs[name][:, :1, :], promoted], dim=1)
            masks = {name: shared_mask for name in masks}

        # Maintain SFTS-style return tuple ordering for make_model.py.
        if 'TIR' in outs:
            result = (outs['RGB'], outs['NIR'], outs['TIR'],
                      (masks['RGB'], masks['NIR'], masks['TIR']))
            if return_scores:
                result += ((scores['RGB'], scores['NIR'], scores['TIR']),)
            return result
        result = (outs['RGB'], outs['NIR'], (masks['RGB'], masks['NIR']))
        if return_scores:
            result += ((scores['RGB'], scores['NIR']),)
        return result
