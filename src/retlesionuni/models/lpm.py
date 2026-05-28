"""Lesion Perception Module (LPM).

Key functions:
1. Attention consistency constraint: L2 loss between attention maps of two augmented views.
2. Spatial attention enhancement: element-wise multiply patch tokens with mean attention weights.

L_lesion = lambda_lesion * ||A(x1) - A(x2)||_2^2
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LesionPerceptionModule(nn.Module):
    """Lesion Perception Module.

    Takes two attention maps from the same image with different augmentations,
    computes a consistency loss (MSE), and enhances features using the mean
    attention weights.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        patch_tokens: torch.Tensor,
        attn1: torch.Tensor,
        attn2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            patch_tokens: (B, N, D) patch tokens from the last ViT layer.
            attn1: (B, num_heads, N+1, N+1) attention map from first augmented view.
            attn2: (B, num_heads, N+1, N+1) attention map from second augmented view.

        Returns:
            (enhanced_features, loss_consistency)
            - enhanced_features: (B, N, D) attention-weighted patch tokens.
            - loss_consistency: scalar MSE loss between the two mean attention maps.
        """
        # Average over heads, remove CLS token
        attn1_avg = self._process_attention(attn1)  # (B, N, N)
        attn2_avg = self._process_attention(attn2)  # (B, N, N)

        # Consistency loss: L2 between averaged attention maps
        loss_consistency = F.mse_loss(attn1_avg, attn2_avg)

        # Mean attention: average over both views, then column-mean gives per-patch weight
        attn_mean = (attn1_avg + attn2_avg) / 2.0  # (B, N, N)
        attn_weights = attn_mean.mean(dim=1)  # (B, N) — average attention received by each patch

        # Enhance features: element-wise multiply patch tokens by attention weights
        enhanced = patch_tokens * attn_weights.unsqueeze(-1)  # (B, N, D)

        return enhanced, loss_consistency

    def _process_attention(self, attn: torch.Tensor) -> torch.Tensor:
        """Process raw attention: mean over heads, remove CLS.

        Args:
            attn: (B, num_heads, N+1, N+1)

        Returns:
            (B, N, N) patch-to-patch attention.
        """
        # Mean over heads
        attn_avg = attn.mean(dim=1)  # (B, N+1, N+1)
        # Remove CLS token
        attn_patch = attn_avg[:, 1:, 1:]  # (B, N, N)
        return attn_patch
