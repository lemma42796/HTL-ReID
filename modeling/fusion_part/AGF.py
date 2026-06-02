import torch
import torch.nn as nn

from ..backbones.vit_pytorch import DropPath
from ..backbones.vit_pytorch import Mlp
from ..backbones.vit_pytorch import trunc_normal_


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.normy = nn.LayerNorm(dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q_ = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_ = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_ = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    @staticmethod
    def _valid_mask(mask, batch_size, num_tokens, device):
        if mask is None:
            return None
        valid = mask.to(device=device, dtype=torch.bool).clone()
        if valid.size(1) != num_tokens:
            raise ValueError('AGF attention mask length does not match patch tokens')
        empty = ~valid.any(dim=1)
        if empty.any():
            valid[empty] = True
        return valid

    def forward(self, x, y, mask=None):
        B, N, C = y.shape
        q = self.q_(x).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_(self.normy(y)).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_(y).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        valid = self._valid_mask(mask, B, N, y.device)
        if valid is not None:
            attn = attn.masked_fill(
                ~valid[:, None, None, :],
                torch.finfo(attn.dtype).min,
            )
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2)
        x = x.reshape(B, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GatedCrossAttn(nn.Module):
    """
    Cross-attention block with the first residual replaced by an adaptive
    gated convex combination (HTL paper §3.4):

        delta = MHA(LN(x), y)
        lam   = sigmoid(Linear([x; delta]))      # per-channel, [B, D]
        x     = (1 - lam) * x + lam * delta
        x     = x + drop_path(MLP(LN(x)))

    The MLP/norm2 residual is kept as a standard pre-norm residual; the
    paper does not specify it but removing it collapses the block into a
    pure linear convex combination with no non-linear capacity.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 gate_init_bias=-2.0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.gate = nn.Linear(2 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        self.reset_gate(gate_init_bias)

    def reset_gate(self, gate_init_bias):
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(gate_init_bias))
        nn.init.zeros_(self.mlp.fc2.weight)
        if self.mlp.fc2.bias is not None:
            nn.init.zeros_(self.mlp.fc2.bias)

    def forward(self, x, y, mask=None):
        delta = self.attn(self.norm1(x), y, mask=mask)
        lam = torch.sigmoid(self.gate(torch.cat([x, delta], dim=-1)))
        x = (1.0 - lam) * x + lam * delta
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ResidualCrossAttn(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, y, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), y, mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BlockRotation(nn.Module):
    def __init__(self, dim, num_heads, mode=0):
        super().__init__()
        self.rotation = ResidualCrossAttn(dim, num_heads, mlp_ratio=4.,
                                          qkv_bias=False, qk_scale=None,
                                          drop=0., attn_drop=0.,
                                          drop_path=0., act_layer=nn.GELU,
                                          norm_layer=nn.LayerNorm)
        self.mode = int(mode)

    def forward(self, x, y, z, masks=None):
        if masks is None:
            masks = (None, None, None)
        mask_x, mask_y, mask_z = masks
        if self.mode == 0:
            x_cls = self.rotation(x[:, 0, :], y[:, 1:, :], mask=mask_y)
            y_cls = self.rotation(y[:, 0, :], z[:, 1:, :], mask=mask_z)
            z_cls = self.rotation(z[:, 0, :], x[:, 1:, :], mask=mask_x)
            x = torch.cat([x_cls.unsqueeze(1), x[:, 1:, :]], dim=1)
            y = torch.cat([y_cls.unsqueeze(1), y[:, 1:, :]], dim=1)
            z = torch.cat([z_cls.unsqueeze(1), z[:, 1:, :]], dim=1)
            return x, y, z
        x_cls = self.rotation(x[:, 0, :], x[:, 1:, :], mask=mask_x)
        y_cls = self.rotation(y[:, 0, :], y[:, 1:, :], mask=mask_y)
        z_cls = self.rotation(z[:, 0, :], z[:, 1:, :], mask=mask_z)
        return torch.cat([x_cls, y_cls, z_cls], dim=-1)


class AGF(nn.Module):
    """
    Quality-aware graph fusion for nighttime RGB/NIR/TIR ReID.

    The default graph mode keeps the original adaptive neighbor fusion. TPM-lite
    replaces the free neighbor choice with fixed cyclic token permutation:
    RGB cls reads NIR patches, NIR cls reads TIR patches, and TIR cls reads
    RGB patches, then the source is rotated in the following steps.
    TPM mode follows TOP-ReID's residual rotation block more closely and is
    intended as an independent auxiliary descriptor branch.
    The final [B, 3D] descriptor preserves the original AGF output contract.
    """

    def __init__(self, dim, num_heads, gate_init_bias=-2.0, quality_scale=True,
                 mode='graph', tpm_steps=3, use_masks=True):
        super().__init__()
        self.quality_scale = bool(quality_scale)
        self.use_masks = bool(use_masks)
        self.mode = mode.lower()
        if self.mode not in ('graph', 'tpm_lite', 'tpm'):
            raise ValueError("MODEL.AGF_MODE must be 'graph', 'tpm_lite', or 'tpm'")
        self.tpm_steps = max(1, int(tpm_steps))
        if self.mode == 'tpm_lite':
            self.tpm_blocks = nn.ModuleList([
                GatedCrossAttn(dim, num_heads, mlp_ratio=4., qkv_bias=False,
                               qk_scale=None, drop=0., attn_drop=0.,
                               drop_path=0., act_layer=nn.GELU,
                               norm_layer=nn.LayerNorm,
                               gate_init_bias=gate_init_bias)
                for _ in range(self.tpm_steps)
            ])
        elif self.mode == 'tpm':
            self.tpm_start = BlockRotation(dim, num_heads, mode=0)
            self.tpm_middle = BlockRotation(dim, num_heads, mode=0)
            self.tpm_end = BlockRotation(dim, num_heads, mode=1)
        else:
            self.cross = GatedCrossAttn(dim, num_heads, mlp_ratio=4., qkv_bias=False,
                                        qk_scale=None, drop=0., attn_drop=0.,
                                        drop_path=0., act_layer=nn.GELU,
                                        norm_layer=nn.LayerNorm,
                                        gate_init_bias=gate_init_bias)
            self.self_aggr = GatedCrossAttn(dim, num_heads, mlp_ratio=4., qkv_bias=False,
                                            qk_scale=None, drop=0., attn_drop=0.,
                                            drop_path=0., act_layer=nn.GELU,
                                            norm_layer=nn.LayerNorm,
                                            gate_init_bias=gate_init_bias)
            hidden = max(dim // 4, 64)
            self.edge_gate = nn.Sequential(
                nn.LayerNorm(2 * dim + 2),
                nn.Linear(2 * dim + 2, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
        self.apply(self._init_weights)
        if self.mode == 'tpm_lite':
            for block in self.tpm_blocks:
                block.reset_gate(gate_init_bias)
        elif self.mode == 'graph':
            self.cross.reset_gate(gate_init_bias)
            self.self_aggr.reset_gate(gate_init_bias)
            nn.init.zeros_(self.edge_gate[-1].weight)
            nn.init.zeros_(self.edge_gate[-1].bias)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _quality_tensor(x, quality_scores):
        if quality_scores is None:
            return torch.ones(x.size(0), 3, device=x.device, dtype=x.dtype)
        return quality_scores.to(device=x.device, dtype=x.dtype).clamp(0.0, 1.0)

    def _edge_weights(self, cls_tokens, quality, target_idx):
        src_indices = [i for i in range(3) if i != target_idx]
        target = cls_tokens[target_idx]
        q_t = quality[:, target_idx:target_idx + 1]
        logits = []
        valid = []
        for src_idx in src_indices:
            source = cls_tokens[src_idx]
            q_s = quality[:, src_idx:src_idx + 1]
            edge_in = torch.cat([target, source, q_t, q_s], dim=-1)
            logit = self.edge_gate(edge_in).squeeze(-1)
            logit = logit + torch.log(q_s.squeeze(-1).clamp_min(1e-6))
            logits.append(logit)
            valid.append(q_s.squeeze(-1) > 0)

        logits = torch.stack(logits, dim=1)
        valid = torch.stack(valid, dim=1)
        very_neg = torch.finfo(logits.dtype).min
        logits = torch.where(valid, logits, torch.full_like(logits, very_neg))
        all_invalid = ~valid.any(dim=1, keepdim=True)
        logits = torch.where(all_invalid, torch.zeros_like(logits), logits)
        weights = torch.softmax(logits, dim=1) * valid.to(logits.dtype)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
        return src_indices, weights

    def _fuse_node(self, feats, cls_tokens, quality, target_idx):
        target_feat = feats[target_idx]
        target_cls = cls_tokens[target_idx]
        self_cls = self.self_aggr(target_cls, target_feat[:, 1:, :])
        src_indices, weights = self._edge_weights(cls_tokens, quality, target_idx)

        cross_cls = torch.zeros_like(self_cls)
        other_quality = torch.zeros_like(quality[:, target_idx])
        valid_count = torch.zeros_like(quality[:, target_idx])
        for pos, src_idx in enumerate(src_indices):
            src_feat = feats[src_idx]
            candidate = self.cross(target_cls, src_feat[:, 1:, :])
            w = weights[:, pos:pos + 1]
            cross_cls = cross_cls + w * candidate
            is_valid = (quality[:, src_idx] > 0).to(quality.dtype)
            other_quality = other_quality + quality[:, src_idx] * is_valid
            valid_count = valid_count + is_valid

        other_quality = other_quality / valid_count.clamp_min(1.0)
        q_t = quality[:, target_idx]
        cross_mix = other_quality / (q_t + other_quality + 1e-6)
        fused = (1.0 - cross_mix.unsqueeze(-1)) * self_cls + cross_mix.unsqueeze(-1) * cross_cls
        if self.quality_scale:
            scale = q_t
        else:
            scale = (q_t > 0).to(fused.dtype)
        return fused * scale.unsqueeze(-1)

    @staticmethod
    def _mask_tuple(masks):
        if masks is None:
            return (None, None, None)
        if len(masks) == 2:
            return (masks[0], masks[1], None)
        return masks

    def _tpm_lite(self, feats, cls_tokens, quality, masks=None):
        masks = self._mask_tuple(masks)
        cls_tokens = list(cls_tokens)
        for step, block in enumerate(self.tpm_blocks):
            source_indices = [
                (target_idx + step + 1) % 3
                for target_idx in range(3)
            ]
            next_cls = []
            for target_idx, src_idx in enumerate(source_indices):
                src_mask = masks[src_idx] if self.use_masks else None
                candidate = block(cls_tokens[target_idx], feats[src_idx][:, 1:, :],
                                  mask=src_mask)
                src_quality = quality[:, src_idx:src_idx + 1]
                next_cls.append(
                    cls_tokens[target_idx] +
                    src_quality * (candidate - cls_tokens[target_idx])
                )
            cls_tokens = next_cls

        if self.quality_scale:
            cls_tokens = [
                cls_tokens[i] * quality[:, i:i + 1]
                for i in range(3)
            ]
        return torch.cat(cls_tokens, dim=-1)

    def _tpm(self, feats, quality, masks=None):
        masks = self._mask_tuple(masks)
        if not self.use_masks:
            masks = (None, None, None)
        x, y, z = feats
        x, y, z = self.tpm_start(x, y, z, masks=masks)
        x, z, y = self.tpm_middle(x, z, y, masks=(masks[0], masks[2], masks[1]))
        cls = self.tpm_end(x, y, z, masks=masks)
        if self.quality_scale:
            cls_nodes = cls.chunk(3, dim=-1)
            cls = torch.cat([
                cls_nodes[i] * quality[:, i:i + 1]
                for i in range(3)
            ], dim=-1)
        return cls

    def forward(self, x, y, z, quality_scores=None, masks=None):
        feats = [x, y, z]
        cls_tokens = [feat[:, 0, :] for feat in feats]
        quality = self._quality_tensor(x, quality_scores)
        if self.mode == 'tpm_lite':
            return self._tpm_lite(feats, cls_tokens, quality, masks=masks)
        if self.mode == 'tpm':
            return self._tpm(feats, quality, masks=masks)
        fused = [
            self._fuse_node(feats, cls_tokens, quality, target_idx=i)
            for i in range(3)
        ]
        return torch.cat(fused, dim=-1)
