"""Asymmetric Loss for multi-label classification (Ridnik et al., ICCV 2021).

L_ASL = -(1/N) * sum_i sum_j [y_ij * (1-p_ij)^gamma_pos * log(p_ij)
                             + (1-y_ij) * p_ij^gamma_neg * log(1-p_ij)]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification.

    Applies different focusing parameters to positive and negative samples
    to handle class imbalance.

    Args:
        gamma_pos: Focusing parameter for positive samples (default 1.0).
        gamma_neg: Focusing parameter for negative samples (default 4.0).
        reduction: 'mean' or 'sum'.
    """

    def __init__(self, gamma_pos: float = 1.0, gamma_neg: float = 4.0, reduction: str = "mean"):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute asymmetric loss.

        Args:
            pred: (B, C) predicted probabilities in [0, 1].
            target: (B, C) binary ground truth labels.

        Returns:
            Scalar loss tensor.
        """
        eps = 1e-8
        pred = torch.clamp(pred, min=eps, max=1.0 - eps)

        pos_term = target * ((1 - pred) ** self.gamma_pos) * torch.log(pred)
        neg_term = (1 - target) * (pred ** self.gamma_neg) * torch.log(1 - pred)

        loss = -(pos_term + neg_term)

        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()
