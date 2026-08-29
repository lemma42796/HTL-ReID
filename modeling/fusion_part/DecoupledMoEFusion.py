"""Lightweight seven-route decoupled mixture-of-experts fusion.

The module uses three modality-specific routes, three pairwise-shared routes,
and one route shared by all modalities while keeping the existing shared
ImageNet ViT backbone and ACI descriptor intact.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualExpert(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        update = self.dropout(self.act(self.proj(self.norm(x))))
        return x + self.scale.tanh() * update


class DecoupledMoEFusion(nn.Module):
    """Create and dynamically gate seven heterogeneous modality routes.

    Historical configurations keep the original gated-concatenation output.
    ``utility_fusion`` instead produces one true mixture and exposes a
    training-only counterfactual objective: a route should receive a high gate
    value when removing it worsens the batch-hard retrieval margin.
    """

    route_names = ('rgb', 'nir', 'tir', 'rgb_nir', 'rgb_tir', 'nir_tir', 'all')

    def __init__(self, dim=768, num_heads=12, gate_heads=4, dropout=0.1,
                 utility_fusion=False, utility_temperature=1.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError('feature dimension must be divisible by attention heads')
        if dim % gate_heads != 0:
            raise ValueError('feature dimension must be divisible by gate heads')
        self.dim = int(dim)
        self.gate_heads = int(gate_heads)
        self.gate_head_dim = self.dim // self.gate_heads
        self.utility_fusion = bool(utility_fusion)
        self.utility_temperature = float(utility_temperature)
        if self.utility_temperature <= 0.0:
            raise ValueError('utility temperature must be positive')

        self.route_tokens = nn.Parameter(torch.empty(7, 1, self.dim))
        nn.init.normal_(self.route_tokens, std=self.dim ** -0.5)
        self.route_attn = nn.ModuleList([
            nn.MultiheadAttention(
                self.dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(7)
        ])
        self.route_norm = nn.ModuleList([nn.LayerNorm(self.dim) for _ in range(7)])
        self.route_attn_scale = nn.Parameter(torch.full((7,), 0.1))
        self.experts = nn.ModuleList([
            _ResidualExpert(self.dim, dropout) for _ in range(7)
        ])

        self.gate_input_norm = nn.LayerNorm(7 * self.dim)
        self.gate_query = nn.Linear(7 * self.dim, self.dim)
        self.gate_key = nn.Linear(self.dim, self.dim, bias=False)
        nn.init.zeros_(self.gate_query.weight)
        nn.init.zeros_(self.gate_query.bias)
        nn.init.eye_(self.gate_key.weight)
        self._last_gate = None
        self._last_utility_target = None

    @property
    def output_dim(self):
        return self.dim if self.utility_fusion else 7 * self.dim

    @staticmethod
    def _route_inputs(rgb, nir, tir):
        rgb_cls, nir_cls, tir_cls = rgb[:, 0], nir[:, 0], tir[:, 0]
        contexts = (
            rgb,
            nir,
            tir,
            torch.cat((rgb, nir), dim=1),
            torch.cat((rgb, tir), dim=1),
            torch.cat((nir, tir), dim=1),
            torch.cat((rgb, nir, tir), dim=1),
        )
        anchors = (
            rgb_cls,
            nir_cls,
            tir_cls,
            0.5 * (rgb_cls + nir_cls),
            0.5 * (rgb_cls + tir_cls),
            0.5 * (nir_cls + tir_cls),
            (rgb_cls + nir_cls + tir_cls) / 3.0,
        )
        return contexts, anchors

    def _extract_routes(self, rgb, nir, tir):
        contexts, anchors = self._route_inputs(rgb, nir, tir)
        batch = rgb.size(0)
        route_features = []
        for index, (context, anchor) in enumerate(zip(contexts, anchors)):
            query = self.route_tokens[index:index + 1].expand(batch, -1, -1)
            attended = self.route_attn[index](
                query, context, context, need_weights=False)[0].squeeze(1)
            scale = self.route_attn_scale[index].tanh()
            route = self.route_norm[index](anchor + scale * attended)
            route_features.append(self.experts[index](route))
        return torch.stack(route_features, dim=1)

    def _route_gates(self, routes):
        batch = routes.size(0)
        query = self.gate_query(
            self.gate_input_norm(routes.flatten(start_dim=1)))
        query = query.view(batch, self.gate_heads, self.gate_head_dim)
        keys = self.gate_key(routes).view(
            batch, 7, self.gate_heads, self.gate_head_dim)
        logits = torch.einsum('bhd,brhd->bhr', query, keys)
        logits = logits / math.sqrt(self.gate_head_dim)
        gates = logits.softmax(dim=-1)
        self._last_gate = gates.detach()
        return gates

    def _utility_mix(self, routes, gates):
        batch = routes.size(0)
        route_heads = routes.view(
            batch, 7, self.gate_heads, self.gate_head_dim)
        weights = gates.permute(0, 2, 1).unsqueeze(-1)
        return (route_heads * weights).sum(dim=1).reshape(batch, self.dim)

    def forward(self, rgb, nir, tir, return_details=False):
        routes = self._extract_routes(rgb, nir, tir)
        gates = self._route_gates(routes)

        if self.utility_fusion:
            fused = self._utility_mix(routes, gates)
            if return_details:
                return fused, routes, gates
            return fused

        batch = routes.size(0)
        route_heads = routes.view(
            batch, 7, self.gate_heads, self.gate_head_dim)
        gated = route_heads * gates.permute(0, 2, 1).unsqueeze(-1)
        # Preserve the initial descriptor scale when the zero-initialized gate
        # starts uniformly across seven routes.
        gated = gated * 7.0
        descriptor = gated.reshape(batch, self.output_dim)
        if return_details:
            return descriptor, routes, gates
        return descriptor

    @staticmethod
    def _per_anchor_retrieval_loss(features, labels):
        """Return a soft batch-hard triplet loss for every valid anchor."""
        features = F.normalize(features.float(), dim=-1)
        labels = labels.view(-1)
        distance = torch.cdist(features, features, p=2)
        same_identity = labels[:, None].eq(labels[None, :])
        diagonal = torch.eye(
            labels.numel(), device=labels.device, dtype=torch.bool)
        positive = same_identity & ~diagonal
        negative = ~same_identity
        hard_positive = distance.masked_fill(
            ~positive, float('-inf')).amax(dim=1)
        hard_negative = distance.masked_fill(
            ~negative, float('inf')).amin(dim=1)
        valid = torch.isfinite(hard_positive) & torch.isfinite(hard_negative)
        losses = torch.zeros_like(hard_positive)
        losses[valid] = F.softplus(
            hard_positive[valid] - hard_negative[valid])
        return losses, valid

    def counterfactual_utility_loss(self, routes, gates, labels):
        """Teach gates to predict each route's marginal retrieval utility.

        Utility targets are calculated without gradients by removing one route
        at a time. Gradients from the KL objective therefore update the gate
        predictor, while the exact fused retrieval branch trains route content.
        """
        if not self.utility_fusion:
            return routes.new_zeros(())
        if labels is None:
            raise ValueError('counterfactual route utility requires labels')

        with torch.no_grad():
            target_routes = routes.detach().float()
            target_gates = gates.detach().float()
            full_feature = self._utility_mix(target_routes, target_gates)
            full_loss, valid = self._per_anchor_retrieval_loss(
                full_feature, labels)
            counterfactual_losses = []
            for removed_route in range(len(self.route_names)):
                keep = torch.ones_like(target_gates)
                keep[:, :, removed_route] = 0.0
                counterfactual_gate = target_gates * keep
                counterfactual_gate = counterfactual_gate / (
                    counterfactual_gate.sum(dim=-1, keepdim=True).clamp_min(1e-12))
                feature = self._utility_mix(
                    target_routes, counterfactual_gate)
                route_loss, route_valid = self._per_anchor_retrieval_loss(
                    feature, labels)
                valid = valid & route_valid
                counterfactual_losses.append(route_loss)

            utility = torch.stack(counterfactual_losses, dim=1)
            utility = utility - full_loss.unsqueeze(1)
            utility = utility - utility.mean(dim=1, keepdim=True)
            utility = utility / utility.std(
                dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
            target = F.softmax(
                utility / self.utility_temperature, dim=-1)
            uniform = torch.full_like(target, 1.0 / len(self.route_names))
            target = torch.where(valid.unsqueeze(1), target, uniform)
            self._last_utility_target = target.detach()

        predicted = gates.float().mean(dim=1).clamp_min(1e-8)
        return F.kl_div(predicted.log(), target, reduction='batchmean')

    def gate_statistics(self):
        if self._last_gate is None:
            return {}
        mean_gate = self._last_gate.float().mean(dim=(0, 1))
        return {
            name: mean_gate[index]
            for index, name in enumerate(self.route_names)
        }

    def utility_statistics(self):
        if self._last_utility_target is None:
            return {}
        mean_target = self._last_utility_target.float().mean(dim=0)
        return {
            name: mean_target[index]
            for index, name in enumerate(self.route_names)
        }
