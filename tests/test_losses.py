"""Unit tests for RetLesionUni loss functions."""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

import torch
from retlesionuni.losses.asl_loss import AsymmetricLoss
from retlesionuni.losses.dice_loss import DiceCELoss
from retlesionuni.losses.contrastive_loss import SupervisedContrastiveLoss


def test_asl_loss_perfect():
    """Perfect predictions should give near-zero ASL loss."""
    loss_fn = AsymmetricLoss(gamma_pos=1.0, gamma_neg=4.0)
    pred = torch.tensor([[0.99, 0.01], [0.01, 0.99]])
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    loss = loss_fn(pred, target)
    assert loss.item() < 0.1, f"Expected low loss for perfect pred, got {loss.item():.4f}"
    print("PASS: asl_loss perfect prediction")


def test_asl_loss_random():
    """Random predictions should give positive ASL loss."""
    loss_fn = AsymmetricLoss()
    pred = torch.sigmoid(torch.randn(8, 8))
    target = (torch.rand(8, 8) > 0.5).float()
    loss = loss_fn(pred, target)
    assert loss.item() > 0, f"Expected positive loss, got {loss.item():.4f}"
    print("PASS: asl_loss random prediction")


def test_dice_ce_loss_perfect():
    """Perfect segmentation should give near-zero Dice+CE loss."""
    loss_fn = DiceCELoss(dice_weight=0.7, ce_weight=0.3)
    # Create logits where correct class has high value
    B, C, H, W = 2, 5, 32, 32
    target = torch.randint(0, C, (B, H, W))
    logits = torch.zeros(B, C, H, W)
    for b in range(B):
        for c in range(C):
            logits[b, c] = 10.0 * (target[b] == c).float()
    loss = loss_fn(logits, target)
    assert loss.item() < 0.5, f"Expected low loss for perfect seg, got {loss.item():.4f}"
    print("PASS: dice_ce_loss perfect segmentation")


def test_dice_ce_loss_random():
    """Bad segmentation should give positive loss."""
    loss_fn = DiceCELoss()
    logits = torch.randn(2, 5, 32, 32)
    target = torch.randint(0, 5, (2, 32, 32))
    loss = loss_fn(logits, target)
    assert loss.item() > 0, f"Expected positive loss, got {loss.item():.4f}"
    print("PASS: dice_ce_loss random segmentation")


def test_contrastive_loss_no_positives():
    """When no DR-positive samples, loss should be 0."""
    loss_fn = SupervisedContrastiveLoss(temperature=0.07)
    z_odir = F.normalize(torch.randn(8, 256), dim=-1)
    z_ddr = F.normalize(torch.randn(8, 256), dim=-1)
    dr_labels = torch.zeros(8)  # All negative
    loss = loss_fn(z_odir, z_ddr, dr_labels)
    assert loss.item() == 0.0, f"Expected 0 loss when no positives, got {loss.item()}"
    print("PASS: contrastive_loss no positives -> zero")


def test_contrastive_loss_with_positives():
    """With DR-positive samples, loss should be positive."""
    loss_fn = SupervisedContrastiveLoss(temperature=0.07)
    z_odir = F.normalize(torch.randn(8, 256), dim=-1)
    z_ddr = F.normalize(torch.randn(8, 256), dim=-1)
    dr_labels = torch.tensor([1, 0, 1, 0, 1, 0, 0, 0])  # Some positive
    loss = loss_fn(z_odir, z_ddr, dr_labels)
    assert loss.item() > 0, f"Expected positive loss with positives, got {loss.item():.4f}"
    print("PASS: contrastive_loss with positives -> positive value")


if __name__ == "__main__":
    import torch.nn.functional as F
    test_asl_loss_perfect()
    test_asl_loss_random()
    test_dice_ce_loss_perfect()
    test_dice_ce_loss_random()
    test_contrastive_loss_no_positives()
    test_contrastive_loss_with_positives()
    print("\nAll loss function tests passed!")
