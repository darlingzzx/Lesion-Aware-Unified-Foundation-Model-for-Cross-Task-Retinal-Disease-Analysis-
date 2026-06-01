"""Prediction heads for RetLesionUni: ODIR classifier and DDR segmentator."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ODIRClassifier(nn.Module):
    """Multi-label classification head for ODIR (8 classes).

    2-layer MLP with BatchNorm + ReLU + Dropout.
    Input: fused cls_token + patch_pooled = 2048-dim (CTAM fused additively before this).
    """

    def __init__(self, dim_in: int = 2048, num_classes: int = 8, hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, dim_in) fused feature vector.

        Returns:
            (B, num_classes) sigmoid probabilities.
        """
        return torch.sigmoid(self.mlp(x))


class DDRSegmentator(nn.Module):
    """Progressive decoder for DDR lesion segmentation (5 classes: bg + EX/HE/MA/SE).

    Architecture (simplified progressive FPN):
      1. FPN: lateral (1×1 conv) + upsample to 64×64 + concat + fusion conv
      2. Progressive upsampling with conv refinement: 64→128→256→512
         Each stage: Conv3×3+BN+ReLU → bilinear 2x upsample
      3. Final 1×1 conv → num_classes

    This replaces the old single 8x bilinear upsampling (which lost small lesions)
    with learned progressive refinement, while keeping the proven FPN structure.
    """

    def __init__(
        self,
        dim_in: int = 1024,
        num_classes: int = 5,
        fpn_dim: int = 256,
        patch_size: int = 16,
        multilevel_layers: List[int] | None = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.multilevel_layers = multilevel_layers or [12, 18, 24]
        num_levels = len(self.multilevel_layers)

        # Lateral convolutions: 1x1 conv to align channel dimensions
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(dim_in, fpn_dim, kernel_size=1) for _ in range(num_levels)
        ])

        # FPN fusion at 64×64: merge multi-level features
        self.fpn_fusion = nn.Sequential(
            nn.Conv2d(fpn_dim * num_levels, fpn_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True),
        )

        # Progressive upsampling decoder: 64→128→256→512
        # Each stage refines features with a conv block before upsampling
        self.decode1 = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(fpn_dim // 2),
            nn.ReLU(inplace=True),
        )  # 64→64 refine, then bilinear 2x → 128

        self.decode2 = nn.Sequential(
            nn.Conv2d(fpn_dim // 2, fpn_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(fpn_dim // 4),
            nn.ReLU(inplace=True),
        )  # 128→128 refine, then bilinear 2x → 256

        self.decode3 = nn.Sequential(
            nn.Conv2d(fpn_dim // 4, fpn_dim // 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(fpn_dim // 8),
            nn.ReLU(inplace=True),
        )  # 256→256 refine, then bilinear 2x → 512

        # Final 1×1 conv → class logits
        self.final_conv = nn.Conv2d(fpn_dim // 8, num_classes, kernel_size=1)

    def forward(self, patch_tokens_list: List[torch.Tensor], img_size: int = 512) -> torch.Tensor:
        """Forward pass with progressive decoding.

        Args:
            patch_tokens_list: List of (B, N, dim_in) patch tokens from different ViT layers.
            img_size: Target output spatial size.

        Returns:
            (B, num_classes, img_size, img_size) logits.
        """
        grid_size = img_size // self.patch_size  # e.g., 512/16 = 32

        # Stage 1: FPN — lateral convs + upsample to 64×64 + concat
        features: list = []
        for i, tokens in enumerate(patch_tokens_list):
            B, N, D = tokens.shape
            feat = tokens.transpose(1, 2).reshape(B, D, grid_size, grid_size)
            feat = self.lateral_convs[i](feat)  # (B, fpn_dim, 32, 32)
            feat = F.interpolate(feat, size=(64, 64), mode="bilinear", align_corners=False)
            features.append(feat)

        x = torch.cat(features, dim=1)  # (B, fpn_dim*3, 64, 64)
        x = self.fpn_fusion(x)  # (B, fpn_dim, 64, 64)

        # Stage 2: Progressive decoder 64→128→256→512
        x = self.decode1(x)  # refine at 64
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # →128

        x = self.decode2(x)  # refine at 128
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # →256

        x = self.decode3(x)  # refine at 256
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # →512

        out = self.final_conv(x)  # (B, num_classes, 512, 512)
        if out.shape[-1] != img_size:
            out = F.interpolate(out, size=(img_size, img_size), mode="bilinear", align_corners=False)
        return out
