"""Lesion Perception Module (LPM).

Key functions:
1. Attention sharpening: learnable temperature recovers structure from uniform MAE attention.
2. Attention consistency constraint: symmetric KL divergence between sharpened attention maps.
3. Spatial attention enhancement: element-wise multiply patch tokens with normalized weights.

L_lesion = lambda_lesion * KL_sym(A(x1)_sharp, A(x2)_sharp)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LesionPerceptionModule(nn.Module):
    """Lesion Perception Module.

    Sharpens attention with a learnable temperature, then computes consistency
    loss (symmetric KL divergence default) between two augmented views, and
    enhances features using normalized mean attention weights.

    Key design choices:
    - Learnable temperature: recovers structure from MAE's near-uniform attention.
      T < 1 sharpens, and the model learns T during fine-tuning.
    - KL divergence: ~37x larger signal than L1 on sharpened attention, giving
      meaningful gradient contribution to the total loss.
    - Weight normalization: attention weights normalized to mean=1 so feature
      enhancement preserves magnitude while differentiating patches.
    """

    def __init__(self, loss_type: str = "kl", temperature: float = 0.05):
        """Args:
            loss_type: Type of consistency loss. One of "kl", "l1", "cosine", "mse".
                       Default "kl": symmetric KL divergence, sensitive to sharpened attention.
            temperature: Initial temperature for attention sharpening.
                         < 1.0 sharpens, > 1.0 smoothes. Learnable.
        """
        super().__init__()
        self.loss_type = loss_type
        self.temperature = nn.Parameter(torch.tensor(temperature))

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
            - loss_consistency: scalar consistency loss between the two mean attention maps.
        """
        # Sharpen attention: softmax(log(attn) / T) recovers logits and re-scales.
        # With T < 1, attention becomes peaky → meaningful consistency loss + feature weights.
        attn1 = self._sharpen(attn1)
        attn2 = self._sharpen(attn2)

        # Average over heads, remove CLS token
        attn1_avg = self._process_attention(attn1)  # (B, N, N)
        attn2_avg = self._process_attention(attn2)  # (B, N, N)

        # Consistency loss
        loss_consistency = self._compute_loss(attn1_avg, attn2_avg)

        # Mean attention: average over both views, then column-mean gives per-patch weight
        attn_mean = (attn1_avg + attn2_avg) / 2.0  # (B, N, N)
        attn_weights = attn_mean.mean(dim=1)  # (B, N), ≈ 1/N due to softmax row-sum=1
        # Normalize so mean=1: prevents ~1000× feature attenuation
        attn_weights = attn_weights / (attn_weights.mean(dim=1, keepdim=True) + 1e-8)

        # Enhance features: element-wise multiply patch tokens by attention weights
        enhanced = patch_tokens * attn_weights.unsqueeze(-1)  # (B, N, D)

        return enhanced, loss_consistency

    def _sharpen(self, attn: torch.Tensor) -> torch.Tensor:
        """Re-apply softmax with learnable temperature to sharpen attention.

        softmax(log(attn) / T) recovers the original logits (up to a constant)
        and re-normalizes with temperature T. T < 1 makes attention peakier,
        revealing task-specific structure even from a uniform MAE prior.

        Args:
            attn: (B, num_heads, N+1, N+1) post-softmax attention.

        Returns:
            (B, num_heads, N+1, N+1) sharpened attention.
        """
        log_attn = torch.log(attn + 1e-12)
        return F.softmax(log_attn / self.temperature, dim=-1)

    def _compute_loss(self, attn1: torch.Tensor, attn2: torch.Tensor) -> torch.Tensor:
        """Compute consistency loss between two attention maps.

        Args:
            attn1: (B, N, N) processed attention map.
            attn2: (B, N, N) processed attention map.

        Returns:
            Scalar loss.
        """
        if self.loss_type == "kl":
            # Symmetric KL divergence: sensitive to distribution differences
            # even when attention is sharpened. KL(P||Q) + KL(Q||P).
            # Clamp to >= 0: floating-point can produce tiny negatives (~-1e-8).
            eps = 1e-12
            kl_12 = (attn1 * (torch.log(attn1 + eps) - torch.log(attn2 + eps))).sum(dim=-1).mean()
            kl_21 = (attn2 * (torch.log(attn2 + eps) - torch.log(attn1 + eps))).sum(dim=-1).mean()
            loss = torch.clamp((kl_12 + kl_21) / 2.0, min=0.0)
        elif self.loss_type == "cosine":
            # Cosine distance: 1 - cos_sim ∈ [0, 2]
            # Flatten spatial dims for cosine_similarity
            cos_sim = F.cosine_similarity(
                attn1.flatten(1), attn2.flatten(1), dim=-1
            )  # (B,)
            loss = (1.0 - cos_sim).mean()
        elif self.loss_type == "l1":
            # L1 loss: more sensitive to small differences than MSE
            loss = F.l1_loss(attn1, attn2)
        else:
            # MSE with per-sample sum (fallback, preserves gradient for tiny values)
            loss = F.mse_loss(attn1, attn2, reduction='none')
            loss = loss.sum(dim=(1, 2)).mean()
        return loss

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
