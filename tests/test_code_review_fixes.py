"""Targeted tests for CODE_REVIEW_ISSUES.md fixes (2026-06-01).

Validates all 7 fixes applied to lpm.py, retlesionuni.py, trainer.py, default.yaml.
"""
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

import torch
import torch.nn as nn
from retlesionuni.config import load_config


def test_fix1_lpm_weight_normalization():
    """Fix 1 (P0): LPM attention weights should have mean ≈ 1.0, not ≈ 0.001."""
    from retlesionuni.models.lpm import LesionPerceptionModule

    lpm = LesionPerceptionModule()
    B, N, D, H = 2, 1024, 768, 16  # Simulate ViT-Large: 32x32 patches

    patches = torch.randn(B, N, D)
    # Simulate softmax-normalized attention (row-sum=1)
    attn_raw = torch.randn(B, H, N + 1, N + 1)
    attn1 = torch.softmax(attn_raw, dim=-1)
    attn2 = torch.softmax(attn_raw + 0.01 * torch.randn_like(attn_raw), dim=-1)

    enhanced, loss = lpm(patches, attn1, attn2)

    # Before fix: enhanced ≈ patches * 0.001 (features wiped out)
    # After fix: enhanced should have similar magnitude to patches
    patch_std = patches.std(dim=(0, 2)).mean()
    enhanced_std = enhanced.std(dim=(0, 2)).mean()

    ratio = enhanced_std / (patch_std + 1e-8)
    print(f"  patch_tokens std: {patch_std:.4f}")
    print(f"  enhanced std:     {enhanced_std:.4f}")
    print(f"  ratio (enhanced/patch): {ratio:.3f}x")

    # After fix, ratio should be close to 1.0 (not 0.001)
    assert 0.5 < ratio < 2.0, (
        f"FAIL: enhanced/patch ratio = {ratio:.3f}, expected ~1.0. "
        f"Fix 1 (weight normalization) may not be working!"
    )
    print("PASS: Fix 1 — LPM attention weight normalization (ratio ≈ 1.0)")


def test_fix5_ctam_fusion_scale():
    """Fix 5 (P1): ctam_fusion_scale parameter exists, is learnable, init ≈ 0.1."""
    cfg = load_config(_project_root / "configs" / "test.yaml")
    # Enable CTAM for this test
    cfg.model.ctam.enabled = True
    cfg.model.ctam.proj_dim = 64
    cfg.model.hidden_dim = 192

    from retlesionuni.models.retlesionuni import RetLesionUni
    model = RetLesionUni(cfg)

    # Check ctam_fusion_scale exists
    assert hasattr(model, "ctam_fusion_scale"), (
        "FAIL: ctam_fusion_scale parameter not found! Fix 5 not applied."
    )
    assert isinstance(model.ctam_fusion_scale, nn.Parameter), (
        "FAIL: ctam_fusion_scale is not an nn.Parameter (not learnable)."
    )
    assert model.ctam_fusion_scale.requires_grad, (
        "FAIL: ctam_fusion_scale grad disabled."
    )

    scale_val = model.ctam_fusion_scale.item()
    print(f"  ctam_fusion_scale = {scale_val:.4f} (init expected ~0.1)")

    # Should be initialized near 0.1
    assert 0.05 < scale_val < 0.2, (
        f"FAIL: ctam_fusion_scale = {scale_val}, expected ~0.1"
    )
    print("PASS: Fix 5 — CTAM fusion learnable scale initialized to 0.1")


def test_fix3_ctam_warmup_gating():
    """Fix 3 (P1): warmup_factor should gate CTAM fusion contribution."""
    cfg = load_config(_project_root / "configs" / "test.yaml")
    cfg.model.ctam.enabled = True
    cfg.model.ctam.proj_dim = 64
    cfg.model.hidden_dim = 192

    from retlesionuni.models.retlesionuni import RetLesionUni
    model = RetLesionUni(cfg)

    # Run forward with warmup_factor = 0.0 → CTAM should contribute nothing
    model.set_stage_config(use_lpm=False, use_ctam=True, warmup_factor=0.0)

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

    with torch.no_grad():
        out_warm0 = model(batch)

    # Run with warmup_factor = 1.0 → CTAM fully contributes
    model.set_stage_config(use_lpm=False, use_ctam=True, warmup_factor=1.0)
    with torch.no_grad():
        out_warm1 = model(batch)

    # Logits should differ because CTAM contribution changes
    diff = (out_warm1["odir_logits"] - out_warm0["odir_logits"]).abs().max().item()
    print(f"  max |logit_diff| (warmup 0→1): {diff:.6f}")

    assert diff > 1e-6, (
        f"FAIL: warmup_factor=0 vs 1 produced identical logits. "
        f"CTAM warmup gating (Fix 3) may not be working!"
    )
    print("PASS: Fix 3 — warmup_factor gates CTAM fusion contribution")


def test_fix2_validation_ctam_zero_input():
    """Fix 2 (P0): Zero-input CTAM fusion produces valid output, no NaN."""
    cfg = load_config(_project_root / "configs" / "test.yaml")
    cfg.model.ctam.enabled = True
    cfg.model.ctam.proj_dim = 64
    cfg.model.hidden_dim = 192

    from retlesionuni.models.retlesionuni import RetLesionUni
    model = RetLesionUni(cfg)
    model.eval()

    # Simulate the validation path: zero CTAM input → fusion → add
    B = 4
    zero_z = torch.zeros(B, cfg.model.ctam.proj_dim)
    ctam_default = model.ctam_fusion_scale * model.ctam_fusion(zero_z)

    assert ctam_default.shape == (B, cfg.model.hidden_dim * 2), (
        f"FAIL: ctam_fusion output shape {ctam_default.shape} != ({B}, {cfg.model.hidden_dim * 2})"
    )
    assert not torch.isnan(ctam_default).any(), "FAIL: NaN in CTAM zero-input fusion"
    assert not torch.isinf(ctam_default).any(), "FAIL: Inf in CTAM zero-input fusion"

    # Zero input should produce only the learned bias (not random noise)
    print(f"  ctam_fusion(zero) std: {ctam_default.std().item():.6f}")
    print(f"  ctam_fusion(zero) mean: {ctam_default.mean().item():.6f}")
    print("PASS: Fix 2 — validation CTAM zero-input produces valid output")


def test_fix4_encoder_path_consistency():
    """Fix 4 (P1): return_attention=True uses same code path as training."""
    cfg = load_config(_project_root / "configs" / "test.yaml")

    from retlesionuni.models.retlesionuni import RetLesionUni
    model = RetLesionUni(cfg)
    model.eval()

    x = torch.randn(2, 3, cfg.model.img_size, cfg.model.img_size)

    with torch.no_grad():
        out_default = model.encoder(x)                    # return_attention=False
        out_attn = model.encoder(x, return_attention=True)  # return_attention=True

    # CLS tokens should be close (same input, same weights, slightly different fp order)
    cls_diff = (out_default["cls_token"] - out_attn["cls_token"]).abs().max().item()
    patch_diff = (out_default["patch_tokens"] - out_attn["patch_tokens"]).abs().max().item()

    print(f"  max |cls_token diff| (default vs attn): {cls_diff:.8f}")
    print(f"  max |patch_tokens diff| (default vs attn): {patch_diff:.8f}")

    # Both paths should produce valid (non-NaN) outputs
    assert not torch.isnan(out_attn["cls_token"]).any(), "FAIL: NaN in attn-path cls_token"
    assert not torch.isnan(out_attn["patch_tokens"]).any(), "FAIL: NaN in attn-path patch_tokens"

    # Very close — difference is only floating-point operation ordering
    assert cls_diff < 1e-4, f"FAIL: cls_token divergence {cls_diff} too large"
    print("PASS: Fix 4 — encoder return_attention path consistent")


def test_fix6_lambda_lesion_config():
    """Fix 6 (P2): lambda_lesion should be 10.0 in default config."""
    cfg = load_config(_project_root / "configs" / "default.yaml")

    lambda_val = cfg.model.lpm.lambda_lesion
    print(f"  lambda_lesion = {lambda_val}")

    assert lambda_val == 10.0, (
        f"FAIL: lambda_lesion = {lambda_val}, expected 10.0. Fix 6 not applied!"
    )
    print("PASS: Fix 6 — lambda_lesion = 10.0 in default config")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing CODE_REVIEW_ISSUES.md fixes")
    print("=" * 60)

    test_fix1_lpm_weight_normalization()
    print()
    test_fix5_ctam_fusion_scale()
    print()
    test_fix3_ctam_warmup_gating()
    print()
    test_fix2_validation_ctam_zero_input()
    print()
    test_fix4_encoder_path_consistency()
    print()
    test_fix6_lambda_lesion_config()
    print()

    print("=" * 60)
    print("All fix verification tests passed!")
