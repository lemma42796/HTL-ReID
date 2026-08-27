"""Training-only cross-modal token reconstruction for the shared backbone.

This module is inspired by TOP-ReID's Complementary Reconstruction Module
(CRM), but is deliberately adapted to HTL-ReID's shared-backbone setting.  A
single modality-conditioned predictor reconstructs one target spectrum from
the other two spectra.  The target tokens are stop-gradient teacher features,
and the module is never used to form the inference descriptor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedCrossModalTokenReconstruction(nn.Module):
    """Reconstruct one modality's patch tokens from the other two modalities.

    The predictor is shared for RGB, NIR, and TIR.  Source and target modality
    embeddings preserve spectral identity without introducing three separate
    reconstruction branches.  Sources are fused patch-by-patch because the
    RGBNT inputs are spatially paired; this keeps the training-only head small
    enough for the project's fixed 20-epoch/30-minute experiment protocol.
    """

    NUM_MODALITIES = 3

    def __init__(self, dim, hidden_dim=256, target_seed=0,
                 all_targets=False, smooth_l1_weight=0.0):
        super().__init__()
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.all_targets = bool(all_targets)
        self.smooth_l1_weight = float(smooth_l1_weight)
        if self.dim <= 0 or self.hidden_dim <= 0:
            raise ValueError(
                'cross-modal reconstruction dimensions must be positive')
        if self.smooth_l1_weight < 0.0:
            raise ValueError(
                'cross-modal reconstruction Smooth L1 weight must be non-negative')

        self.modality_embedding = nn.Parameter(
            torch.empty(self.NUM_MODALITIES, self.hidden_dim))
        self.source_norm = nn.LayerNorm(self.dim)
        self.source_projection = nn.Linear(self.dim, self.hidden_dim)
        self.source_fusion = nn.Sequential(
            nn.LayerNorm(2 * self.hidden_dim),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.predictor = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.dim),
        )
        nn.init.normal_(self.modality_embedding, std=0.02)
        self._target_generator = torch.Generator(device='cpu')
        self._target_generator.manual_seed(int(target_seed))
        self._target_history = []
        self._last_target_index = None
        self._last_loss = None

    def _validate_features(self, features):
        if len(features) != self.NUM_MODALITIES:
            raise ValueError(
                'cross-modal reconstruction requires RGB, NIR, and TIR tokens')
        reference_shape = features[0].shape
        if len(reference_shape) != 3 or reference_shape[1] < 2:
            raise ValueError(
                'reconstruction features must have shape [batch, tokens, dim]')
        if reference_shape[-1] != self.dim:
            raise ValueError('reconstruction feature dimension mismatch')
        if any(feature.shape != reference_shape for feature in features[1:]):
            raise ValueError(
                'all reconstruction modalities must share the same token shape')

    def predict(self, features, target_index):
        """Predict target patch tokens without reading the target modality."""
        self._validate_features(features)
        target_index = int(target_index)
        if not 0 <= target_index < self.NUM_MODALITIES:
            raise ValueError('target modality index must be 0, 1, or 2')

        source_indices = [
            index for index in range(self.NUM_MODALITIES)
            if index != target_index
        ]
        source_tokens = []
        for source_index in source_indices:
            patches = features[source_index][:, 1:, :]
            projected = self.source_projection(self.source_norm(patches))
            source_tokens.append(
                projected + self.modality_embedding[source_index].view(1, 1, -1))

        context = self.source_fusion(torch.cat(source_tokens, dim=-1))
        context = context + self.modality_embedding[target_index].view(1, 1, -1)
        return self.predictor(context)

    def _target_loss(self, features, target_index):
        """Return scale-stable token reconstruction loss for one target."""
        prediction = self.predict(features, target_index)
        teacher = features[target_index][:, 1:, :].detach()
        # Compute similarities in fp32 under AMP. Normalizing both terms keeps
        # the optional Smooth L1 component commensurate with cosine distance.
        prediction = F.normalize(prediction.float(), dim=-1, eps=1e-6)
        teacher = F.normalize(teacher.float(), dim=-1, eps=1e-6)
        cosine = 1.0 - (prediction * teacher).sum(dim=-1).mean()
        if self.smooth_l1_weight == 0.0:
            return cosine
        regression = F.smooth_l1_loss(prediction, teacher)
        return cosine + self.smooth_l1_weight * regression

    def forward(self, features, target_index=None):
        """Return reconstruction loss for one target or all three targets."""
        self._validate_features(features)
        if target_index is not None:
            target_indices = (int(target_index),)
        elif self.all_targets:
            target_indices = tuple(range(self.NUM_MODALITIES))
        else:
            target_index = int(torch.randint(
                self.NUM_MODALITIES, (1,),
                generator=self._target_generator).item())
            target_indices = (target_index,)

        losses = [self._target_loss(features, index) for index in target_indices]
        loss = torch.stack(losses).mean()
        self._target_history.extend(target_indices)
        self._last_target_index = (target_indices[0] if len(target_indices) == 1
                                   else tuple(target_indices))
        self._last_loss = loss.detach()
        return loss

    def target_history(self):
        """Return the deterministic training target sequence for tracing."""
        return tuple(self._target_history)
