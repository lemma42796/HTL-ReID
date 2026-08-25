"""Paper-faithful salient and frequency token selection from EDITOR.

The implementation keeps the selection rule from the official EDITOR source,
but removes visualization-only code and CUDA hard-coding so it can be used by
HTL-ReID on any device. BCC and OCFR are intentionally not part of this module.
"""

import torch
import torch.nn as nn


class PartAttention(nn.Module):
    """Roll out ViT attention and union the per-head top-ratio patches."""

    def __init__(self, ratio=0.5):
        super().__init__()
        if not 0.0 < ratio <= 1.0:
            raise ValueError("SFTS ratio must be in (0, 1]")
        self.ratio = float(ratio)

    @torch.no_grad()
    def forward(self, attn_list):
        if not attn_list:
            raise ValueError("SFTS requires a non-empty backbone attention list")

        # EDITOR forms A_L @ ... @ A_1 and consumes only its CLS row. Propagate
        # that row backwards through the same matrices instead of materializing
        # every full M x M product. This is mathematically identical while
        # reducing rollout complexity from O(M^3) to O(M^2). Selection ends in
        # hard top-k indices, so constructing an autograd graph here cannot
        # provide a gradient and only retains large intermediate tensors.
        cls_attention = attn_list[-1][:, :, 0, :]
        for attention in reversed(attn_list[:-1]):
            cls_attention = torch.matmul(
                cls_attention.unsqueeze(-2), attention
            ).squeeze(-2)
        cls_attention = cls_attention[:, :, 1:]
        batch_size, num_heads, num_patches = cls_attention.shape
        keep = max(1, int(num_patches * self.ratio))
        mask = torch.zeros(
            batch_size, num_patches, dtype=torch.bool,
            device=cls_attention.device,
        )
        # Union all per-head selections in one scatter instead of launching a
        # separate top-k/scatter pair for each attention head.
        indices = cls_attention.topk(keep, dim=2).indices
        mask.scatter_(1, indices.reshape(batch_size, num_heads * keep), True)
        return mask


class SFTS(nn.Module):
    """Select one shared patch set for all available modalities.

    Each modality first produces a per-head attention mask. The masks are
    united across heads and modalities, then united with the optional
    frequency mask exactly as in the official EDITOR implementation.
    """

    def __init__(self, ratio=0.5):
        super().__init__()
        self.part_select = PartAttention(ratio=ratio)

    @staticmethod
    def _mask_features(features, mask):
        patches = features[:, 1:, :] * mask.unsqueeze(-1).to(features.dtype)
        return torch.cat((features[:, :1, :], patches), dim=1)

    def forward(self, RGB_feat, RGB_attn, NIR_feat=None, NIR_attn=None,
                TIR_feat=None, TIR_attn=None, img_path=None, writer=None,
                epoch=None, mask_fre=None, quality_scores=None,
                return_scores=False, return_gates=False):
        del img_path, writer, epoch, quality_scores
        if return_scores:
            raise ValueError("SFTS provides a hard mask, not continuous scores")

        modalities = [("RGB", RGB_feat, RGB_attn)]
        if NIR_feat is not None:
            modalities.append(("NIR", NIR_feat, NIR_attn))
        if TIR_feat is not None:
            modalities.append(("TIR", TIR_feat, TIR_attn))

        shared_mask = None
        for _, features, attentions in modalities:
            if attentions is None:
                raise ValueError("SFTS requires attention maps for every modality")
            modality_mask = self.part_select(attentions)
            if modality_mask.shape[1] != features.shape[1] - 1:
                raise ValueError("SFTS attention and feature patch counts differ")
            shared_mask = (modality_mask if shared_mask is None
                           else shared_mask | modality_mask)

        if mask_fre is not None:
            if mask_fre.shape != shared_mask.shape:
                raise ValueError("SFTS frequency mask shape must match patch mask")
            shared_mask = shared_mask | mask_fre.bool()

        selected = {
            name: self._mask_features(features, shared_mask)
            for name, features, _ in modalities
        }
        masks = tuple(shared_mask for _ in modalities)
        result = tuple(selected[name] for name, _, _ in modalities) + (masks,)

        # Preserve the hard-gate portion of the selector contract used by FACR.
        if return_gates:
            gate = shared_mask.to(RGB_feat.dtype)
            result += (tuple(gate for _ in modalities),)
        return result
