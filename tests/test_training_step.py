"""Integration test: single training step end-to-end with real preprocessed data."""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

import torch
from retlesionuni.config import load_config
from retlesionuni.data.transforms import (
    get_ddr_transform,
    get_odir_eval_transform,
    get_odir_transforms,
)
from retlesionuni.data.odir_dataset import ODIRDataset
from retlesionuni.data.ddr_dataset import DDRDataset
from retlesionuni.data.joint_loader import JointDataLoader
from retlesionuni.losses.asl_loss import AsymmetricLoss
from retlesionuni.losses.dice_loss import DiceCELoss
from retlesionuni.models.retlesionuni import RetLesionUni


def test_data_loading():
    """Verify datasets load correctly from preprocessed cache."""
    cfg = load_config(_project_root / "configs" / "test.yaml")
    cache = cfg.data.preprocessed_cache
    img_size = cfg.model.img_size

    print("Testing ODIR dataset...")
    odir_tr_v1, odir_tr_v2 = get_odir_transforms(img_size)
    odir_cache = str(Path(cache) / "odir")
    odir_ds = ODIRDataset(odir_cache, "train", odir_tr_v1, odir_tr_v2)
    assert len(odir_ds) == 4900, f"Expected 4900 ODIR train samples, got {len(odir_ds)}"
    sample = odir_ds[0]
    assert sample["image_v1"].shape == (3, img_size, img_size)
    assert sample["image_v2"].shape == (3, img_size, img_size)
    assert sample["labels"].shape == (8,)
    assert sample["dr_label"] in (0, 1)
    print("  ODIR: {} samples, image {}, labels {}".format(
        len(odir_ds), sample["image_v1"].shape, sample["labels"].shape))

    print("Testing DDR dataset...")
    ddr_tr = get_ddr_transform(img_size, is_train=True)
    ddr_cache = str(Path(cache) / "ddr")
    ddr_ds = DDRDataset(ddr_cache, "train", ddr_tr)
    assert len(ddr_ds) == 383, f"Expected 383 DDR train samples, got {len(ddr_ds)}"
    sample2 = ddr_ds[0]
    assert sample2["image"].shape == (3, img_size, img_size)
    assert sample2["mask"].shape == (img_size, img_size)
    print("  DDR: {} samples, image {}, mask {}".format(
        len(ddr_ds), sample2["image"].shape, sample2["mask"].shape))

    print("PASS: data loading")


def test_joint_loader():
    """Verify joint loader yields correctly structured batches."""
    cfg = load_config(_project_root / "configs" / "test.yaml")
    cache = cfg.data.preprocessed_cache
    img_size = cfg.model.img_size

    odir_tr_v1, odir_tr_v2 = get_odir_transforms(img_size)
    ddr_tr = get_ddr_transform(img_size, is_train=True)
    odir_ds = ODIRDataset(str(Path(cache) / "odir"), "train", odir_tr_v1, odir_tr_v2)
    ddr_ds = DDRDataset(str(Path(cache) / "ddr"), "train", ddr_tr)

    loader = JointDataLoader(odir_ds, ddr_ds, batch_size_odir=2, batch_size_ddr=2, num_workers=0, pin_memory=False)

    batch = next(iter(loader))
    assert "odir" in batch and "ddr" in batch
    assert batch["odir"]["image_v1"].shape == (2, 3, img_size, img_size)
    assert batch["ddr"]["image"].shape == (2, 3, img_size, img_size)
    print("  Joint batch: odir={}, ddr={}".format(
        batch["odir"]["image_v1"].shape, batch["ddr"]["image"].shape))
    print("PASS: joint loader")


def test_training_step():
    """Run a single forward + backward + optimizer step, verify no NaN."""
    cfg = load_config(_project_root / "configs" / "test.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))

    # Create model
    model = RetLesionUni(cfg).to(device)
    model.set_stage_config(use_lpm=False, use_ctam=False)

    # Create losses
    loss_odir = AsymmetricLoss(gamma_pos=1.0, gamma_neg=4.0)
    loss_ddr = DiceCELoss(dice_weight=0.7, ce_weight=0.3)

    # Create a dummy batch mimicking the joint loader format
    img_size = cfg.model.img_size
    batch = {
        "odir": {
            "image_v1": torch.randn(2, 3, img_size, img_size).to(device),
            "image_v2": torch.randn(2, 3, img_size, img_size).to(device),
            "labels": torch.randint(0, 2, (2, 8)).float().to(device),
            "dr_label": torch.tensor([1, 0]).to(device),
        },
        "ddr": {
            "image": torch.randn(2, 3, img_size, img_size).to(device),
            "mask": torch.randint(0, 5, (2, img_size, img_size)).to(device),
        },
    }

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )

    # Single training step
    model.train()
    optimizer.zero_grad()

    outputs = model(batch)
    l_odir = loss_odir(outputs["odir_logits"], batch["odir"]["labels"])
    l_ddr = loss_ddr(outputs["ddr_seg"], batch["ddr"]["mask"])
    loss = l_odir + l_ddr + outputs["loss_lesion"] + outputs["loss_align"]

    loss.backward()
    optimizer.step()

    print("  Loss_O={:.4f}, Loss_D={:.4f}, Loss_les={:.4f}, Loss_align={:.4f}".format(
        l_odir.item(), l_ddr.item(),
        outputs["loss_lesion"].item(), outputs["loss_align"].item()))

    # Check no NaN
    assert not torch.isnan(loss), "Loss is NaN!"
    assert not any(torch.isnan(p).any() for p in model.parameters() if p.requires_grad), "NaN in gradients!"

    print("PASS: training step (no NaN)")


if __name__ == "__main__":
    test_data_loading()
    print()
    test_joint_loader()
    print()
    test_training_step()
    print()
    print("All integration tests passed!")
