"""Supervised contrastive alignment loss for CTAM.

For each DDR sample j, computes InfoNCE over all ODIR samples,
weighted by DR-positive labels:
    positive pairs: DR-positive ODIR <-> DDR (semantically consistent)
    negative pairs: DR-negative ODIR <-> DDR (semantically inconsistent)

L_align = (1/|P|) * sum_{i in P} -(1/B_D) * sum_j
    log[ exp(sim(Z_O[i], Z_D[j])/tau) / sum_k exp(sim(Z_O[k], Z_D[j])/tau) ]
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SupervisedContrastiveLoss(nn.Module):
    """Supervised contrastive loss using DR labels as positive/negative indicator.

    Args:
        temperature: Temperature parameter tau (default 0.07).
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_odir: torch.Tensor,
        z_ddr: torch.Tensor,
        dr_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute supervised contrastive loss.

        Args:
            z_odir: (B_O, D) L2-normalized ODIR features.
            z_ddr: (B_D, D) L2-normalized DDR features.
            dr_labels: (B_O,) binary DR labels (1 = positive, 0 = negative).

        Returns:
            Scalar loss. Returns 0 if no DR-positive samples in batch.
        """
        B_O, B_D = z_odir.shape[0], z_ddr.shape[0]
        pos_mask = (dr_labels == 1)

        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=z_odir.device)

        # Similarity matrix: (B_O, B_D)
        sim = torch.matmul(z_odir, z_ddr.T) / self.temperature

        loss = 0.0
        for j in range(B_D):
            pos_sim = sim[pos_mask, j]       # (num_pos,)
            all_sim = sim[:, j]               # (B_O,)

            numerator = torch.exp(pos_sim).sum()
            denominator = torch.exp(all_sim).sum()

            loss += -torch.log(numerator / (denominator + 1e-8))

        return loss / B_D
