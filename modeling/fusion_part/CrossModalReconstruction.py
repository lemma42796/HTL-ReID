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

    def __init__(self, dim, hidden_dim=256):
        super().__init__()
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        if self.dim <= 0 or self.hidden_dim <= 0:
            raise ValueError(
                'cross-modal reconstruction dimensions must be positive')

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

    def forward(self, features, target_index=None):
        """Return normalized cosine reconstruction loss for one target."""
        self._validate_features(features)
        if target_index is None:
            # The global torch RNG is already seeded by the formal runner, so
            # this per-batch target choice remains reproducible.
            target_index = int(torch.randint(self.NUM_MODALITIES, (1,)).item())
        target_index = int(target_index)

        prediction = self.predict(features, target_index)
        teacher = features[target_index][:, 1:, :].detach()
        # Compute the similarity in fp32 under AMP.  Casting does not interrupt
        # gradients from the predictor and source modalities.
        prediction = F.normalize(prediction.float(), dim=-1, eps=1e-6)
        teacher = F.normalize(teacher.float(), dim=-1, eps=1e-6)
        loss = 1.0 - (prediction * teacher).sum(dim=-1).mean()
        self._last_target_index = target_index
        self._last_loss = loss.detach()
        return loss
