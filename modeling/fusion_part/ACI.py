"""Cross-modal token fusion modules.

``ACI`` (Adaptive Cross-modal Interaction) is the project extension of
TOP-ReID's fixed cyclic Token Permutation Module: every target modality's
class token adaptively aggregates evidence from the complete patch sequences
of the other modalities, with sample-conditioned routing weights and gates.
"""

import torch
import torch.nn as nn

from ..backbones.vit_pytorch import DropPath, Mlp, trunc_normal_


class ScoreBiasedCrossAttention(nn.Module):
    """Class-token query over a source modality's complete patch sequence."""

    def __init__(self, dim, num_heads=12, score_bias_scale=0.25,
                 score_floor=0.05, detach_scores=True, qkv_bias=False, attn_drop=0.0,
                 proj_drop=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError('fusion dimension must be divisible by num_heads')
        self.num_heads = int(num_heads)
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.score_bias_scale = float(score_bias_scale)
        if self.score_bias_scale < 0.0:
            raise ValueError('score_bias_scale must be non-negative')
        self.score_floor = float(score_floor)
        if not 0.0 <= self.score_floor < 1.0:
            raise ValueError('score_floor must be in [0, 1)')
        # Start exactly from score-free interaction (T2). The bounded gain can
        # only introduce selector guidance gradually when the identification
        # objective finds it useful.
        self.score_bias_gain = nn.Parameter(
            torch.zeros(()), requires_grad=self.score_bias_scale != 0.0)
        self.detach_scores = bool(detach_scores)
        self.source_norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    @staticmethod
    def _normalize_scores(scores):
        lo = scores.min(dim=1, keepdim=True).values
        hi = scores.max(dim=1, keepdim=True).values
        return (scores - lo) / (hi - lo + 1e-6)

    def forward(self, query, source_tokens, scores=None, mask=None):
        batch, tokens, dim = source_tokens.shape
        if query.shape != (batch, dim):
            raise ValueError('query and source token dimensions do not match')

        source = self.source_norm(source_tokens)
        q = self.q(query).view(batch, self.num_heads, 1, self.head_dim)
        k = self.k(source).view(batch, tokens, self.num_heads, self.head_dim)
        k = k.permute(0, 2, 1, 3)
        v = self.v(source_tokens).view(batch, tokens, self.num_heads, self.head_dim)
        v = v.permute(0, 2, 1, 3)
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        attention_gate = None
        if mask is not None:
            if mask.shape != (batch, tokens):
                raise ValueError('selection mask shape must match source patches')
            attention_gate = mask.to(device=logits.device, dtype=logits.dtype)
            if not attention_gate.detach().bool().any(dim=1).all():
                raise ValueError('selection mask must retain at least one source patch')

        if scores is not None and self.score_bias_scale != 0.0:
            if scores.shape != (batch, tokens):
                raise ValueError('selection score shape must match source patches')
            if self.detach_scores:
                scores = scores.detach()
            scores = self._normalize_scores(scores.to(dtype=logits.dtype))
            # Centered, bounded residual guidance replaces the former log
            # prior, which imposed up to a 20x attention ratio at every stage.
            scores = (scores - self.score_floor).clamp_min(0.0)
            scores = scores / (1.0 - self.score_floor)
            bias = 2.0 * scores - 1.0
            gain = self.score_bias_scale * torch.tanh(self.score_bias_gain)
            logits = logits + gain * bias[:, None, None, :]

        attention = logits.softmax(dim=-1)
        if attention_gate is not None:
            attention = attention * attention_gate[:, None, None, :]
            attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        attention = self.attn_drop(attention)
        context = torch.matmul(attention, v).transpose(1, 2).reshape(batch, dim)
        return self.proj_drop(self.proj(context))


class ResidualCrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention followed by a standard MLP residual."""

    def __init__(self, dim, num_heads=12, score_bias_scale=0.0,
                 score_floor=0.05, detach_scores=True, mlp_ratio=4.0,
                 drop_path=0.0):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.attn = ScoreBiasedCrossAttention(
            dim, num_heads=num_heads, score_bias_scale=score_bias_scale,
            score_floor=score_floor, detach_scores=detach_scores)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = Mlp(
            in_features=dim, hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU, drop=0.0)

    def forward(self, query, source_tokens, scores=None, mask=None):
        query = query + self.drop_path(
            self.attn(
                self.query_norm(query), source_tokens,
                scores=scores, mask=mask))
        return query + self.drop_path(self.mlp(self.mlp_norm(query)))


class AdaptiveRoutingStage(nn.Module):
    """All-connected, sample-adaptive cross-modal class-token update."""

    def __init__(self, dim, num_heads, score_bias_scale, score_floor,
                 detach_scores, gate_init_bias=0.0):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.attn = ScoreBiasedCrossAttention(
            dim, num_heads=num_heads, score_bias_scale=score_bias_scale,
            score_floor=score_floor, detach_scores=detach_scores)
        self.route = nn.Sequential(
            nn.LayerNorm(2 * dim + 2),
            nn.Linear(2 * dim + 2, max(dim // 4, 64)),
            nn.GELU(),
            nn.Linear(max(dim // 4, 64), 1),
        )
        self.gate = nn.Linear(2 * dim, dim)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = Mlp(
            in_features=dim, hidden_features=4 * dim,
            act_layer=nn.GELU, drop=0.0)
        nn.init.zeros_(self.route[-1].weight)
        nn.init.zeros_(self.route[-1].bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(gate_init_bias))
        self._last_route_weights = None

    @staticmethod
    def _score_stats(scores, cls):
        if scores is None:
            return torch.ones(cls.size(0), 2, device=cls.device, dtype=cls.dtype)
        score = scores.to(device=cls.device, dtype=cls.dtype)
        return torch.stack([score.mean(dim=1), score.amax(dim=1)], dim=-1)

    @staticmethod
    def _replace_cls(feat, cls):
        return torch.cat([cls.unsqueeze(1), feat[:, 1:, :]], dim=1)

    def _update_one(self, target_idx, feats, scores, masks):
        target = feats[target_idx][:, 0, :]
        contexts = []
        route_logits = []
        for source_idx, source in enumerate(feats):
            if source_idx == target_idx:
                continue
            source_score = None if scores is None else scores[source_idx]
            source_mask = None if masks is None else masks[source_idx]
            if source_score is not None and self.attn.detach_scores:
                source_score = source_score.detach()
            contexts.append(self.attn(
                self.query_norm(target), source[:, 1:, :],
                scores=source_score, mask=source_mask))
            route_input = torch.cat([
                target, source[:, 0, :], self._score_stats(source_score, target)
            ], dim=-1)
            route_logits.append(self.route(route_input))

        route_weight = torch.softmax(torch.cat(route_logits, dim=1), dim=1)
        context_stack = torch.stack(contexts, dim=1)
        context = (context_stack * route_weight.unsqueeze(-1)).sum(dim=1)
        gate = torch.sigmoid(self.gate(torch.cat([target, context], dim=-1)))
        updated = target + gate * context
        return updated + self.mlp(self.mlp_norm(updated)), route_weight

    def forward(self, feats, scores=None, masks=None):
        updated_and_routes = [
            self._update_one(i, feats, scores, masks) for i in range(3)
        ]
        updated = [item[0] for item in updated_and_routes]
        self._last_route_weights = torch.stack(
            [item[1] for item in updated_and_routes], dim=1)
        return tuple(
            self._replace_cls(feat, cls)
            for feat, cls in zip(feats, updated)
        )


class IndependentMaskedAggregation(nn.Module):
    """Refine each modality CLS from its own selected patch tokens.

    This is the independent stage that precedes collaborative ACI routing in
    T11. One block is shared across modalities so the added capacity does not
    encode modality-specific parameter-count differences. Patch sequences are
    preserved; only the class tokens are updated.
    """

    def __init__(self, dim, num_heads=12):
        super().__init__()
        self.block = ResidualCrossAttentionBlock(
            dim, num_heads=num_heads, score_bias_scale=0.0)

    @staticmethod
    def _replace_cls(feat, cls):
        return torch.cat([cls.unsqueeze(1), feat[:, 1:, :]], dim=1)

    def forward(self, feats, masks):
        if masks is None or len(masks) != len(feats):
            raise ValueError(
                'independent masked aggregation requires one mask per modality')
        refined = []
        for feat, mask in zip(feats, masks):
            cls = self.block(
                feat[:, 0, :], feat[:, 1:, :], mask=mask)
            refined.append(self._replace_cls(feat, cls))
        return tuple(refined)


class FinalSelfRefinement(nn.Module):
    """Let an adaptively routed CLS finally recover its own local evidence."""

    def __init__(self, dim, num_heads, scale_init=0.1):
        super().__init__()
        if float(scale_init) < 0.0:
            raise ValueError('ACI self-refinement scale must be non-negative')
        self.query_norm = nn.LayerNorm(dim)
        self.attn = ScoreBiasedCrossAttention(
            dim, num_heads=num_heads, score_bias_scale=0.0)
        self.residual_scale = nn.Parameter(
            torch.full((dim,), float(scale_init)))

    @staticmethod
    def _replace_cls(feat, cls):
        return torch.cat([cls.unsqueeze(1), feat[:, 1:, :]], dim=1)

    def forward(self, feat, mask=None, residual_token=None):
        target = feat[:, 0, :]
        source_tokens = feat[:, 1:, :]
        attention_mask = mask
        if residual_token is not None:
            if residual_token.shape != target.shape:
                raise ValueError(
                    'HS residual token must match the ACI class-token shape')
            source_tokens = torch.cat(
                [source_tokens, residual_token.unsqueeze(1)], dim=1)
            if attention_mask is None:
                attention_mask = torch.ones(
                    target.size(0), source_tokens.size(1),
                    device=target.device, dtype=target.dtype)
            else:
                summary_gate = torch.ones(
                    target.size(0), 1,
                    device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat(
                    [attention_mask, summary_gate], dim=1)

        context = self.attn(
            self.query_norm(target), source_tokens, mask=attention_mask)
        refined = target + self.residual_scale.to(context.dtype) * context
        return self._replace_cls(feat, refined)


class ACI(nn.Module):
    """Adaptive Cross-modal Interaction.

    Unlike the fixed-cycle TOP-ReID TPM it replaces, every target modality
    reads both other modalities. Continuous selector scores can bias patch
    attention, while a learned route and per-channel residual gate control
    source and injection strength per sample. An optional independent
    pre-stage first lets each class token read its own masked patches before
    collaborative interaction. An optional final refinement lets each routed
    class token read its own selected patches and the HS residual summary.
    """

    def __init__(self, dim, num_heads=12, steps=3, score_bias_scale=0.25,
                 score_floor=0.05, detach_scores=True, gate_init_bias=0.0,
                 route_balance_weight=0.0, self_refine=False,
                 self_refine_scale_init=0.1,
                 independent_aggregation=False):
        super().__init__()
        self.route_balance_weight = float(route_balance_weight)
        self.self_refine_enabled = bool(self_refine)
        self.independent_aggregation_enabled = bool(independent_aggregation)
        if self.route_balance_weight < 0.0:
            raise ValueError('ACI route balance weight must be non-negative')
        if self.independent_aggregation_enabled:
            self.independent_aggregation = IndependentMaskedAggregation(
                dim, num_heads=num_heads)
        self.stages = nn.ModuleList([
            AdaptiveRoutingStage(
                dim, num_heads=num_heads,
                score_bias_scale=score_bias_scale,
                score_floor=score_floor,
                detach_scores=detach_scores,
                gate_init_bias=gate_init_bias)
            for _ in range(max(1, int(steps)))
        ])
        if self.self_refine_enabled:
            self.self_refinement = FinalSelfRefinement(
                dim, num_heads=num_heads,
                scale_init=self_refine_scale_init)
        self.apply(ACI._init_weights)
        # Restore neutral initial routing/gating after generic initialization.
        for stage in self.stages:
            nn.init.zeros_(stage.route[-1].weight)
            nn.init.zeros_(stage.route[-1].bias)
            nn.init.zeros_(stage.gate.weight)
            nn.init.constant_(stage.gate.bias, float(gate_init_bias))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, rgb, nir, tir, scores=None, masks=None,
                residual_tokens=None):
        if scores is not None and len(scores) != 3:
            raise ValueError('ACI requires one score tensor per modality')
        if masks is not None and len(masks) != 3:
            raise ValueError('ACI requires one selection mask tensor per modality')
        if residual_tokens is not None and len(residual_tokens) != 3:
            raise ValueError('ACI requires one HS residual token per modality')
        if residual_tokens is not None and not self.self_refine_enabled:
            raise ValueError('HS residual tokens require ACI self-refinement')
        feats = (rgb, nir, tir)
        if self.independent_aggregation_enabled:
            feats = self.independent_aggregation(feats, masks)
        for stage in self.stages:
            feats = stage(feats, scores=scores, masks=masks)
            if self.route_balance_weight == 0.0:
                stage._last_route_weights = stage._last_route_weights.detach()
        if self.self_refine_enabled:
            masks = masks or (None, None, None)
            residual_tokens = residual_tokens or (None, None, None)
            feats = tuple(
                self.self_refinement(feat, mask=mask, residual_token=residual)
                for feat, mask, residual in zip(
                    feats, masks, residual_tokens)
            )
        return torch.cat([feat[:, 0, :] for feat in feats], dim=-1)

    def regularization_loss(self, reference):
        """Prevent batch-level source starvation while preserving per-sample routing."""
        if self.route_balance_weight == 0.0:
            return torch.zeros((), device=reference.device, dtype=reference.dtype)
        route_weights = [
            stage._last_route_weights for stage in self.stages
            if stage._last_route_weights is not None
        ]
        if not route_weights:
            return torch.zeros((), device=reference.device, dtype=reference.dtype)
        losses = []
        for weights in route_weights:
            mean_route = weights.float().mean(dim=0)
            losses.append((mean_route - 0.5).pow(2).mean())
        loss = torch.stack(losses).mean().to(dtype=reference.dtype)
        return self.route_balance_weight * loss

    def route_statistics(self):
        """Return detached aggregate diagnostics for the latest forward pass."""
        route_weights = [
            stage._last_route_weights.detach().float() for stage in self.stages
            if stage._last_route_weights is not None
        ]
        if not route_weights:
            return {}
        weights = torch.stack(route_weights, dim=0)
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log())
        return {
            'mean_entropy': entropy.sum(dim=-1).mean(),
            'mean_max_probability': weights.amax(dim=-1).mean(),
            'mean_balance_deviation': (
                weights.mean(dim=1) - 0.5
            ).abs().mean(),
        }
