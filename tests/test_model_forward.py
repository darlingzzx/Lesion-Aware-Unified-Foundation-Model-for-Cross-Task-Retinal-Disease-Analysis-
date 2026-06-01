"""Test RetLesionUni model components: forward pass shape verification."""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

import torch
from retlesionuni.config import load_config
from retlesionuni.models.retlesionuni import RetLesionUni


def test_model_instantiation():
    """Verify model can be created with test config."""
    print("Loading test config...")
    cfg = load_config(_project_root / "configs" / "test.yaml")

    print("Creating model...")
    model = RetLesionUni(cfg)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("  Total params: {:,}".format(total_params))
    print("  Trainable: {:,}".format(trainable))
    print("PASS: model instantiation")

    return model, cfg


def test_encoder_forward(model, cfg):
    img_size = cfg.model.img_size
    patch_size = cfg.model.patch_size
    num_patches = (img_size // patch_size) ** 2

    x = torch.randn(2, 3, img_size, img_size)

    with torch.no_grad():
        out = model.encoder(x)
        assert out["cls_token"].shape == (2, cfg.model.hidden_dim)
        assert out["patch_tokens"].shape == (2, num_patches, cfg.model.hidden_dim)
        print("  Basic forward: CLS {}, patches {}".format(out["cls_token"].shape, out["patch_tokens"].shape))

        out2 = model.encoder(x, return_multilevel=True)
        n_levels = len(cfg.model.multilevel_layers)
        assert len(out2["multilevel_features"]) == n_levels
        print("  Multilevel: {} levels OK".format(n_levels))

        out3 = model.encoder(x, return_attention=True)
        attn = out3["last_attention"]
        expected_attn = (2, cfg.model.num_heads, num_patches + 1, num_patches + 1)
        assert attn.shape == expected_attn, "Attn shape {} != {}".format(attn.shape, expected_attn)
        print("  Attention: {}".format(attn.shape))

    print("PASS: encoder forward")


def test_heads_forward(model, cfg):
    hidden_dim = cfg.model.hidden_dim
    odir_classes = cfg.model.odir_num_classes
    ddr_classes = cfg.model.ddr_num_classes
    img_size = cfg.model.img_size
    num_patches = (img_size // cfg.model.patch_size) ** 2

    # ODIR input is cls_token + patch_pooled = hidden_dim * 2
    fused_features = torch.randn(4, hidden_dim * 2)
    odir_out = model.odir_head(fused_features)
    assert odir_out.shape == (4, odir_classes)
    assert (odir_out >= 0).all() and (odir_out <= 1).all()
    print("  ODIR head: {} in [0,1]".format(odir_out.shape))

    patch_tokens = [torch.randn(2, num_patches, hidden_dim) for _ in range(3)]
    ddr_out = model.ddr_head(patch_tokens, img_size)
    assert ddr_out.shape == (2, ddr_classes, img_size, img_size), \
        f"DDR shape {ddr_out.shape} != (2, {ddr_classes}, {img_size}, {img_size})"
    print("  DDR head: {}".format(ddr_out.shape))

    print("PASS: heads forward")


def test_lpm_forward():
    from retlesionuni.models.lpm import LesionPerceptionModule

    lpm = LesionPerceptionModule()
    B, N, D, H = 2, 196, 768, 12
    patches = torch.randn(B, N, D)
    # ViT attention is always post-softmax (positive, row-sum=1)
    attn1 = torch.softmax(torch.randn(B, H, N + 1, N + 1), dim=-1)
    attn2 = torch.softmax(torch.randn(B, H, N + 1, N + 1), dim=-1)

    enhanced, loss = lpm(patches, attn1, attn2)
    assert enhanced.shape == (B, N, D)
    assert loss.item() >= 0
    print("  LPM: enhanced {}, loss={:.6f}".format(enhanced.shape, loss.item()))

    _, loss0 = lpm(patches, attn1, attn1)
    assert loss0.item() < 1e-6
    print("  LPM identical: loss={:.10f}".format(loss0.item()))

    print("PASS: LPM forward")


def test_ctam_forward():
    from retlesionuni.models.ctam import CrossTaskAlignmentModule

    ctam = CrossTaskAlignmentModule(dim_in=512, dim_proj=128, num_heads=4)
    B_O, B_D = 8, 8
    f_odir = torch.randn(B_O, 512)
    f_ddr = torch.randn(B_D, 512)
    dr_labels = torch.tensor([1, 0, 1, 0, 1, 0, 0, 0])

    z_odir_enh, z_ddr, loss = ctam(f_odir, f_ddr, dr_labels)
    assert z_odir_enh.shape == (B_O, 128)
    assert z_ddr.shape == (B_D, 128)
    assert loss.item() > 0
    print("  CTAM: z_odir {}, z_ddr {}, loss={:.4f}".format(z_odir_enh.shape, z_ddr.shape, loss.item()))

    dr_labels_neg = torch.zeros(B_O)
    _, _, loss0 = ctam(f_odir, f_ddr, dr_labels_neg)
    assert loss0.item() == 0.0
    print("  CTAM no positives: loss={}".format(loss0.item()))

    print("PASS: CTAM forward")


def test_full_model_forward(model, cfg):
    img_size = cfg.model.img_size

    batch = {
        "odir": {
            "image_v1": torch.randn(2, 3, img_size, img_size),
            "image_v2": torch.randn(2, 3, img_size, img_size),
            "labels": torch.randint(0, 2, (2, 8)).float(),
            "dr_label": torch.tensor([1, 0]),
        },
        "ddr": {
            "image": torch.randn(2, 3, img_size, img_size),
            "mask": torch.randint(0, 5, (2, img_size, img_size)),
        },
    }

    model.set_stage_config(use_lpm=False, use_ctam=False)
    with torch.no_grad():
        out1 = model(batch)
    assert out1["odir_logits"].shape == (2, cfg.model.odir_num_classes)
    assert out1["ddr_seg"].shape == (2, cfg.model.ddr_num_classes, img_size, img_size)
    print("  Stage1: odir {}, ddr {}".format(out1["odir_logits"].shape, out1["ddr_seg"].shape))
    print("  Stage1 losses: lesion={:.4f}, align={:.4f}".format(
        out1["loss_lesion"].item(), out1["loss_align"].item()))

    if model.lpm is not None and model.ctam is not None:
        model.set_stage_config(use_lpm=True, use_ctam=True, warmup_factor=0.5)
        with torch.no_grad():
            out2 = model(batch)
        print("  Stage2 losses: lesion={:.4f}, align={:.4f}".format(
            out2["loss_lesion"].item(), out2["loss_align"].item()))

    print("PASS: full model forward")


if __name__ == "__main__":
    print("=" * 60)
    model, cfg = test_model_instantiation()
    print()
    test_encoder_forward(model, cfg)
    print()
    test_heads_forward(model, cfg)
    print()
    test_lpm_forward()
    print()
    test_ctam_forward()
    print()
    test_full_model_forward(model, cfg)
    print()
    print("=" * 60)
    print("All model tests passed!")
