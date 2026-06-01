"""Dice + Cross-Entropy loss for multi-class segmentation.

L_DDR = 0.7 * L_Dice + 0.3 * L_CE
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCELoss(nn.Module):
    """Combined Dice and Cross-Entropy loss for segmentation.

    Args:
        dice_weight: Weight for Dice loss component (default 0.7).
        ce_weight: Weight for CE loss component (default 0.3).
        smooth: Smoothing factor for Dice (default 1.0).
        ignore_index: Class index to ignore in CE computation (default -1 = none).
        class_weights: Optional list/tensor of per-class weights for CE loss.
    """

    def __init__(
        self,
        dice_weight: float = 0.7,
        ce_weight: float = 0.3,
        smooth: float = 1.0,
        ignore_index: int = -1,
        class_weights: list | None = None,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth
        self.ignore_index = ignore_index
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Dice+CE loss.

        Args:
            pred: (B, C, H, W) logits (before softmax).
            target: (B, H, W) integer class indices.

        Returns:
            Scalar loss tensor.
        """
        # Ensure target is Long (int64), cross_entropy requires Long
        target = target.long()

        # CE loss (with optional class weights for imbalanced data)
        ce_weight = self.class_weights
        if ce_weight is not None and ce_weight.device != pred.device:
            ce_weight = ce_weight.to(pred.device)
        loss_ce = F.cross_entropy(
            pred, target,
            weight=ce_weight,
            ignore_index=self.ignore_index,
        )

        # Per-class Dice loss
        num_classes = pred.shape[1]
        pred_soft = F.softmax(pred, dim=1)
        loss_dice = 0.0

        for c in range(num_classes):
            pred_c = pred_soft[:, c, ...]
            target_c = (target == c).float()

            intersection = (pred_c * target_c).sum(dim=(1, 2))
            union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))

            dice_per_sample = (2.0 * intersection + self.smooth) / (union + self.smooth)
            loss_dice += (1.0 - dice_per_sample).mean()

        loss_dice /= num_classes

        return self.dice_weight * loss_dice + self.ce_weight * loss_ce


class SoftDiceLoss(nn.Module):
    """Multi-class soft Dice loss only (for evaluation metrics)."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = pred.shape[1]
        pred_soft = F.softmax(pred, dim=1)
        loss = 0.0
        for c in range(num_classes):
            pred_c = pred_soft[:, c, ...]
            target_c = (target == c).float()
            intersection = (pred_c * target_c).sum(dim=(1, 2))
            union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))
            dice_per_sample = (2.0 * intersection + self.smooth) / (union + self.smooth)
            loss += (1.0 - dice_per_sample).mean()
        return loss / num_classes
