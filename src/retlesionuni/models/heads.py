"""Prediction heads for RetLesionUni: ODIR classifier and DDR segmentator."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ODIRClassifier(nn.Module):
    """Multi-label classification head for ODIR (8 classes).

    h_ODIR = W * z_cls + b
    y_hat = sigmoid(h_ODIR)
    """

    def __init__(self, dim_in: int = 1024, num_classes: int = 8):
        super().__init__()
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            cls_token: (B, dim_in) CLS token from ViT encoder.

        Returns:
            (B, num_classes) sigmoid probabilities.
        """
        return torch.sigmoid(self.fc(cls_token))


class DDRSegmentator(nn.Module):
    """Multi-level FPN-like segmentation head for DDR (5 classes: bg + EX/HE/MA/SE).

    Takes patch tokens from layers 12, 18, 24 of ViT, applies 1x1 convolutions
    to align channels, upsamples, concatenates, and produces per-pixel predictions.
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

        # Final prediction head
        self.final_conv = nn.Sequential(
            nn.Conv2d(fpn_dim * num_levels, fpn_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_dim, num_classes, kernel_size=1),
        )

    def forward(self, patch_tokens_list: List[torch.Tensor], img_size: int = 512) -> torch.Tensor:
        """Forward pass.

        Args:
            patch_tokens_list: List of (B, N, dim_in) patch tokens from different ViT layers.
            img_size: Target output spatial size.

        Returns:
            (B, num_classes, img_size, img_size) logits.
        """
        grid_size = img_size // self.patch_size  # e.g., 512/16 = 32
        features: list = []

        for i, tokens in enumerate(patch_tokens_list):
            B, N, D = tokens.shape
            # Reshape to spatial: (B, D, H, W)
            feat = tokens.transpose(1, 2).reshape(B, D, grid_size, grid_size)
            # Lateral conv
            feat = self.lateral_convs[i](feat)
            # Upsample to common resolution (128x128)
            feat = F.interpolate(feat, size=(grid_size * 2, grid_size * 2), mode="bilinear", align_corners=False)
            features.append(feat)

        # Concatenate along channel dim
        fused = torch.cat(features, dim=1)  # (B, fpn_dim*3, 2*grid_size, 2*grid_size)

        # Final convs
        out = self.final_conv(fused)  # (B, num_classes, 2*grid_size, 2*grid_size)

        # Upsample to target size
        out = F.interpolate(out, size=(img_size, img_size), mode="bilinear", align_corners=False)

        return out
