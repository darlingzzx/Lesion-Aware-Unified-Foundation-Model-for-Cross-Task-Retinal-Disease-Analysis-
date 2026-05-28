"""Cross-Task Alignment Module (CTAM).

Aligns ODIR and DDR feature spaces using:
1. Separate projection matrices: W_proj^O, W_proj^D (1024 -> 256), not shared.
2. L2 normalization after projection.
3. Cross-attention: ODIR queries DDR features.
4. Supervised contrastive loss using DR labels as positive/negative indicator.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..losses.contrastive_loss import SupervisedContrastiveLoss


class CrossTaskAlignmentModule(nn.Module):
    """Cross-Task Alignment Module.

    Args:
        dim_in: Input feature dimension (default 1024).
        dim_proj: Projection dimension (default 256).
        num_heads: Number of cross-attention heads (default 8).
        alpha: Residual scaling factor (default 0.5).
        temperature: Contrastive loss temperature (default 0.07).
    """

    def __init__(
        self,
        dim_in: int = 1024,
        dim_proj: int = 256,
        num_heads: int = 8,
        alpha: float = 0.5,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.proj_odir = nn.Linear(dim_in, dim_proj)
        self.proj_ddr = nn.Linear(dim_in, dim_proj)
        self.cross_attn = nn.MultiheadAttention(dim_proj, num_heads, batch_first=True)
        self.alpha = alpha
        self.contrastive_loss = SupervisedContrastiveLoss(temperature=temperature)

    def forward(
        self,
        f_odir: torch.Tensor,
        f_ddr: torch.Tensor,
        dr_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            f_odir: (B_O, dim_in) ODIR CLS tokens.
            f_ddr: (B_D, dim_in) DDR CLS tokens.
            dr_labels: (B_O,) binary DR labels (1=positive, 0=negative).

        Returns:
            (z_odir_enhanced, z_ddr, loss_align)
            - z_odir_enhanced: (B_O, dim_proj) cross-attention enhanced ODIR features.
            - z_ddr: (B_D, dim_proj) DDR projected features.
            - loss_align: scalar contrastive alignment loss.
        """
        # Project and L2 normalize
        z_odir = F.normalize(self.proj_odir(f_odir), p=2, dim=-1)  # (B_O, dim_proj)
        z_ddr = F.normalize(self.proj_ddr(f_ddr), p=2, dim=-1)  # (B_D, dim_proj)

        # Cross-attention: ODIR queries DDR
        z_odir_attn, _ = self.cross_attn(
            query=z_odir.unsqueeze(0),
            key=z_ddr.unsqueeze(0),
            value=z_ddr.unsqueeze(0),
        )
        z_odir_attn = z_odir_attn.squeeze(0)

        # Residual connection
        z_odir_enhanced = z_odir + self.alpha * z_odir_attn

        # Contrastive alignment loss
        loss_align = self.contrastive_loss(z_odir, z_ddr, dr_labels)

        return z_odir_enhanced, z_ddr, loss_align
