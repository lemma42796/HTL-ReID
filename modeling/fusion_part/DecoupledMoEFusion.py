"""Lightweight seven-route decoupled mixture-of-experts fusion.

The module uses three modality-specific routes, three pairwise-shared routes,
and one route shared by all modalities while keeping the existing shared
ImageNet ViT backbone and ACI descriptor intact.
"""

import math

import torch
import torch.nn as nn


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
    """Create and dynamically gate seven heterogeneous modality routes."""

    route_names = ('rgb', 'nir', 'tir', 'rgb_nir', 'rgb_tir', 'nir_tir', 'all')

    def __init__(self, dim=768, num_heads=12, gate_heads=4, dropout=0.1):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError('feature dimension must be divisible by attention heads')
        if dim % gate_heads != 0:
            raise ValueError('feature dimension must be divisible by gate heads')
        self.dim = int(dim)
        self.gate_heads = int(gate_heads)
        self.gate_head_dim = self.dim // self.gate_heads

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

    @property
    def output_dim(self):
        return 7 * self.dim

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

    def forward(self, rgb, nir, tir):
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

        routes = torch.stack(route_features, dim=1)
        query = self.gate_query(
            self.gate_input_norm(routes.flatten(start_dim=1)))
        query = query.view(batch, self.gate_heads, self.gate_head_dim)
        keys = self.gate_key(routes).view(
            batch, 7, self.gate_heads, self.gate_head_dim)
        logits = torch.einsum('bhd,brhd->bhr', query, keys)
        logits = logits / math.sqrt(self.gate_head_dim)
        gates = logits.softmax(dim=-1)
        self._last_gate = gates.detach()

        route_heads = routes.view(
            batch, 7, self.gate_heads, self.gate_head_dim)
        gated = route_heads * gates.permute(0, 2, 1).unsqueeze(-1)
        # Preserve the initial descriptor scale when the zero-initialized gate
        # starts uniformly across seven routes.
        gated = gated * 7.0
        return gated.reshape(batch, self.output_dim)

    def gate_statistics(self):
        if self._last_gate is None:
            return {}
        mean_gate = self._last_gate.float().mean(dim=(0, 1))
        return {
            name: mean_gate[index]
            for index, name in enumerate(self.route_names)
        }
