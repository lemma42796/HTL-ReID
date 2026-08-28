"""Hierarchical Token Selection (HS).

Rolls out ViT attention across layers (A_L @ ... @ A_1), keeps the per-head
top-K patches, and unites the masks across heads and modalities. The optional
frequency mask is merged into the same shared union. BCC and OCFR are
intentionally not part of this module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PartAttention(nn.Module):
    """Roll out ViT attention and union the per-head top-ratio patches."""

    def __init__(self, ratio=0.5):
        super().__init__()
        if not 0.0 < ratio <= 1.0:
            raise ValueError("HS ratio must be in (0, 1]")
        self.ratio = float(ratio)

    @staticmethod
    def _rollout_cls(attn_list):
        if not attn_list:
            raise ValueError("HS requires a non-empty backbone attention list")

        # Propagate the CLS row backwards through the attention matrices instead
        # of materializing every full M x M product. This is mathematically
        # identical while
        # reducing rollout complexity from O(M^3) to O(M^2). Selection ends in
        # hard top-k indices, so constructing an autograd graph here cannot
        # provide a gradient and only retains large intermediate tensors.
        with torch.no_grad():
            cls_attention = attn_list[-1][:, :, 0, :]
            for attention in reversed(attn_list[:-1]):
                cls_attention = torch.matmul(
                    cls_attention.unsqueeze(-2), attention
                ).squeeze(-2)
            cls_attention = cls_attention[:, :, 1:]
        return cls_attention

    def forward(self, attn_list, candidate_weights=None, candidates=None):
        cls_attention = self._rollout_cls(attn_list)
        batch_size, num_heads, num_patches = cls_attention.shape
        # Retain a modality-specific saliency distribution for summarizing
        # tokens outside the final shared mask. Selection itself remains the
        # paper-faithful per-head top-K union.
        saliency = cls_attention.mean(dim=1).clamp_min(0.0)

        if candidate_weights is not None:
            if candidates is None or candidate_weights.numel() != len(candidates):
                raise ValueError("HS candidate weights and K candidates differ")
            candidate_k = [min(max(1, int(k)), num_patches) for k in candidates]
            max_keep = max(candidate_k)
            ranked = cls_attention.topk(max_keep, dim=2).indices
            candidate_masks = []
            for keep_k in candidate_k:
                mask = torch.zeros(
                    batch_size, num_heads, num_patches,
                    dtype=cls_attention.dtype, device=cls_attention.device,
                )
                mask.scatter_(2, ranked[:, :, :keep_k], 1.0)
                candidate_masks.append(mask)
            candidate_masks = torch.stack(candidate_masks, dim=0)
            head_gate = torch.einsum(
                'c,cbhn->bhn',
                candidate_weights.to(candidate_masks.dtype), candidate_masks)
            # Differentiable union. With straight-through one-hot candidate
            # weights its forward value is the exact hard head union.
            union_gate = 1.0 - torch.prod(1.0 - head_gate, dim=1)
            return union_gate.detach().bool(), union_gate, saliency

        keep = max(1, int(num_patches * self.ratio))
        mask = torch.zeros(
            batch_size, num_patches, dtype=torch.bool,
            device=cls_attention.device,
        )
        # Union all per-head selections in one scatter instead of launching a
        # separate top-k/scatter pair for each attention head.
        indices = cls_attention.topk(keep, dim=2).indices
        mask.scatter_(1, indices.reshape(batch_size, num_heads * keep), True)
        return mask, saliency


class HS(nn.Module):
    """Select one shared patch set for all available modalities.

    Each modality first produces a per-head attention mask. The masks are
    united across heads and modalities, then united with the optional
    frequency mask. When requested, each modality also summarizes patches
    outside that shared mask into one residual token without changing the
    hard mask.
    """

    def __init__(self, ratio=0.5, learnable_k=False, k_candidates=None,
                 gumbel_tau=1.0, gumbel_tau_min=0.2,
                 gumbel_tau_decay=0.9, budget_loss_weight=0.05):
        super().__init__()
        self.part_select = PartAttention(ratio=ratio)
        self.learnable_k = bool(learnable_k)
        self.k_candidates = tuple(int(k) for k in (k_candidates or (1, 2, 4, 8, 16)))
        if any(k <= 0 for k in self.k_candidates):
            raise ValueError("HS K candidates must be positive")
        if len(set(self.k_candidates)) != len(self.k_candidates):
            raise ValueError("HS K candidates must be unique")
        self.gumbel_tau = float(gumbel_tau)
        self.gumbel_tau_min = float(gumbel_tau_min)
        self.gumbel_tau_decay = float(gumbel_tau_decay)
        self.budget_loss_weight = float(budget_loss_weight)
        if self.learnable_k:
            self.k_logits = nn.Parameter(torch.zeros(len(self.k_candidates)))
        self._last_budget_loss = None
        self._last_logged_epoch = None

    def _candidate_weights(self, epoch):
        if not self.learnable_k:
            return None
        if self.training:
            epoch_index = max(int(epoch or 1) - 1, 0)
            temperature = max(
                self.gumbel_tau_min,
                self.gumbel_tau * (self.gumbel_tau_decay ** epoch_index),
            )
            return F.gumbel_softmax(
                self.k_logits, tau=temperature, hard=True, dim=0)
        index = self.k_logits.argmax()
        return F.one_hot(index, num_classes=len(self.k_candidates)).to(
            dtype=self.k_logits.dtype)

    def regularization_loss(self, reference):
        if self._last_budget_loss is None:
            return torch.zeros((), device=reference.device, dtype=reference.dtype)
        return self._last_budget_loss.to(device=reference.device, dtype=reference.dtype)

    def learned_k_probabilities(self):
        if not self.learnable_k:
            return None
        return torch.softmax(self.k_logits.detach(), dim=0)

    @staticmethod
    def _residual_token(features, shared_mask, saliency):
        """Attention-weighted summary of patches excluded by the union mask."""
        patches = features[:, 1:, :]
        dropped = ~shared_mask
        weights = saliency.to(dtype=patches.dtype) * dropped.to(patches.dtype)
        denominator = weights.sum(dim=1, keepdim=True)
        residual = torch.einsum('bn,bnd->bd', weights, patches)
        residual = residual / denominator.clamp_min(1e-12)
        # A full mask has no residual evidence. Keep the summary exactly zero
        # instead of relying on division by the clamped denominator.
        return torch.where(denominator > 0, residual, torch.zeros_like(residual))

    @staticmethod
    def _mask_features(features, mask):
        patches = features[:, 1:, :] * mask.unsqueeze(-1).to(features.dtype)
        return torch.cat((features[:, :1, :], patches), dim=1)

    def forward(self, RGB_feat, RGB_attn, NIR_feat=None, NIR_attn=None,
                TIR_feat=None, TIR_attn=None, img_path=None, writer=None,
                epoch=None, mask_fre=None, quality_scores=None,
                return_scores=False, return_gates=False,
                return_residual_tokens=False):
        del img_path, quality_scores
        if return_scores:
            raise ValueError("HS provides a hard mask, not continuous scores")

        modalities = [("RGB", RGB_feat, RGB_attn)]
        if NIR_feat is not None:
            modalities.append(("NIR", NIR_feat, NIR_attn))
        if TIR_feat is not None:
            modalities.append(("TIR", TIR_feat, TIR_attn))

        candidate_weights = self._candidate_weights(epoch)
        shared_mask = None
        shared_gate = None
        modality_saliency = []
        for _, features, attentions in modalities:
            if attentions is None:
                raise ValueError("HS requires attention maps for every modality")
            if self.learnable_k:
                modality_mask, modality_gate, saliency = self.part_select(
                    attentions, candidate_weights=candidate_weights,
                    candidates=self.k_candidates)
            else:
                modality_mask, saliency = self.part_select(attentions)
                modality_gate = modality_mask.to(features.dtype)
            if modality_mask.shape[1] != features.shape[1] - 1:
                raise ValueError("HS attention and feature patch counts differ")
            shared_mask = (modality_mask if shared_mask is None
                           else shared_mask | modality_mask)
            shared_gate = (modality_gate if shared_gate is None else
                           1.0 - (1.0 - shared_gate) * (1.0 - modality_gate))
            modality_saliency.append(saliency)

        if mask_fre is not None:
            if mask_fre.shape != shared_mask.shape:
                raise ValueError("HS frequency mask shape must match patch mask")
            shared_mask = shared_mask | mask_fre.bool()
            frequency_gate = mask_fre.to(shared_gate.dtype)
            shared_gate = 1.0 - (1.0 - shared_gate) * (1.0 - frequency_gate)

        if self.learnable_k and self.training:
            candidate_tensor = torch.tensor(
                self.k_candidates, device=candidate_weights.device,
                dtype=candidate_weights.dtype)
            expected_fraction = (
                candidate_weights * candidate_tensor / candidate_tensor.max()
            ).sum()
            self._last_budget_loss = self.budget_loss_weight * (
                0.5 * shared_gate.mean() + 0.5 * expected_fraction)
        else:
            self._last_budget_loss = None

        if (writer is not None and self.learnable_k and epoch is not None and
                self._last_logged_epoch != int(epoch)):
            probabilities = self.learned_k_probabilities()
            expected_k = sum(
                k * probabilities[i] for i, k in enumerate(self.k_candidates))
            writer.add_scalar('HS/expected_k', expected_k.item(), epoch)
            writer.add_scalar('HS/final_union_ratio',
                              shared_mask.float().mean().item(), epoch)
            self._last_logged_epoch = int(epoch)

        selected = {
            name: self._mask_features(features, shared_gate)
            for name, features, _ in modalities
        }
        masks = tuple(shared_mask for _ in modalities)
        result = tuple(selected[name] for name, _, _ in modalities) + (masks,)

        # Preserve the hard-gate portion of the selector contract used by ACI.
        if return_gates:
            gate = shared_gate.to(RGB_feat.dtype)
            result += (tuple(gate for _ in modalities),)
        if return_residual_tokens:
            residual_tokens = tuple(
                self._residual_token(features, shared_mask, saliency)
                for (_, features, _), saliency in zip(
                    modalities, modality_saliency)
            )
            result += (residual_tokens,)
        return result
