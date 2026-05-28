"""Evaluation metrics for RetLesionUni: ODIR classification + DDR segmentation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def compute_odir_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    """Compute ODIR multi-label classification metrics.

    Args:
        preds: (N, 8) sigmoid probabilities in [0, 1].
        targets: (N, 8) binary labels.

    Returns:
        Dict with 'accuracy', 'f1', 'auc', 'per_class_f1'.
    """
    pred_binary = (preds > 0.5).astype(int)

    # Exact match accuracy
    accuracy = (pred_binary == targets).all(axis=1).mean()

    # Macro F1
    f1 = f1_score(targets, pred_binary, average="macro", zero_division=0)

    # Per-class F1
    per_class_f1 = f1_score(targets, pred_binary, average=None, zero_division=0)

    # Macro AUC
    try:
        auc = roc_auc_score(targets, preds, average="macro")
    except ValueError:
        auc = 0.5  # Fallback if only one class present

    return {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "auc": float(auc),
        "per_class_f1": per_class_f1.tolist(),
    }


def compute_ddr_metrics(pred_logits: np.ndarray, targets: np.ndarray) -> dict:
    """Compute DDR segmentation metrics.

    Args:
        pred_logits: (N, 5, H, W) logits or probabilities.
        targets: (N, H, W) integer class labels.

    Returns:
        Dict with 'mIoU', 'mDice', 'per_class_iou', 'per_class_dice'.
    """
    num_classes = pred_logits.shape[1]
    pred_labels = pred_logits.argmax(axis=1)  # (N, H, W)

    iou_per_class = []
    dice_per_class = []

    for c in range(num_classes):
        pred_c = (pred_labels == c)
        target_c = (targets == c)
        intersection = (pred_c & target_c).sum()
        union = (pred_c | target_c).sum()
        total_pred = pred_c.sum()
        total_target = target_c.sum()

        iou = intersection / max(union, 1)
        dice = 2 * intersection / max(total_pred + total_target, 1)

        iou_per_class.append(float(iou))
        dice_per_class.append(float(dice))

    # Mean over all classes
    mIoU = np.mean(iou_per_class)
    mDice = np.mean(dice_per_class)

    # Lesion-only mean (exclude background=0)
    lesion_ious = iou_per_class[1:] if len(iou_per_class) > 1 else iou_per_class
    lesion_dices = dice_per_class[1:] if len(dice_per_class) > 1 else dice_per_class

    return {
        "mIoU": float(mIoU),
        "mDice": float(mDice),
        "lesion_mIoU": float(np.mean(lesion_ious)),
        "lesion_mDice": float(np.mean(lesion_dices)),
        "per_class_iou": iou_per_class,
        "per_class_dice": dice_per_class,
    }
