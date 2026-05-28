"""Explainability analysis: Attention Rollout and Lesion Localization IoU (A_loc)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def attention_rollout(
    attn_matrices: list[torch.Tensor],
    head_fusion: str = "mean",
    discard_ratio: float = 0.0,
) -> torch.Tensor:
    """Compute attention rollout (Abnar & Zuidema, ACL 2020).

    Accumulates attention across layers via matrix multiplication,
    incorporating residual connections.

    Args:
        attn_matrices: List of (B, heads, N+1, N+1) attention matrices per layer.
        head_fusion: How to fuse heads ('mean', 'min', 'max').
        discard_ratio: Fraction of lowest attention values to zero out.

    Returns:
        (B, N, N) rollout attention matrix.
    """
    if head_fusion == "mean":
        attn_stacked = torch.stack([a.mean(dim=1) for a in attn_matrices])  # (L, B, N+1, N+1)
    elif head_fusion == "min":
        attn_stacked = torch.stack([a.min(dim=1).values for a in attn_matrices])
    elif head_fusion == "max":
        attn_stacked = torch.stack([a.max(dim=1).values for a in attn_matrices])
    else:
        raise ValueError(f"Unknown head_fusion: {head_fusion}")

    # Remove CLS token
    attn_stacked = attn_stacked[:, :, 1:, 1:]  # (L, B, N, N)

    # Add residual connection (identity matrix)
    eye = torch.eye(attn_stacked.shape[-1], device=attn_stacked.device)
    attn_residual = 0.5 * attn_stacked + 0.5 * eye

    # Accumulate: multiply sequentially
    L, B, N, _ = attn_residual.shape
    rollout = attn_residual[0]  # (B, N, N)
    for l in range(1, L):
        rollout = torch.matmul(attn_residual[l], rollout)

    return rollout


def compute_lesion_localization(
    attention_map: np.ndarray,
    gt_mask: np.ndarray,
    threshold: float = 0.3,
) -> float:
    """Compute lesion localization IoU (A_loc).

    A_loc = |Binary(A) AND M_lesion| / |Binary(A) OR M_lesion|

    Args:
        attention_map: (H, W) attention map from rollout.
        gt_mask: (H, W) binary lesion mask (union of all lesion classes).
        threshold: Threshold for binarizing the attention map.

    Returns:
        A_loc IoU value in [0, 1].
    """
    # Resize attention map to match mask
    if attention_map.shape != gt_mask.shape:
        attention_map = F.interpolate(
            torch.tensor(attention_map).unsqueeze(0).unsqueeze(0),
            size=gt_mask.shape,
            mode="bilinear",
            align_corners=False,
        ).squeeze().numpy()

    # Normalize attention to [0, 1]
    attn_min, attn_max = attention_map.min(), attention_map.max()
    if attn_max > attn_min:
        attention_map = (attention_map - attn_min) / (attn_max - attn_min)

    # Binarize
    attn_binary = (attention_map > threshold).astype(np.uint8)

    # IoU with lesion mask
    intersection = (attn_binary & gt_mask).sum()
    union = (attn_binary | gt_mask).sum()

    return float(intersection / max(union, 1))
