"""Diagnose two issues: (A) DDR mDice low, (B) L_les near zero.

Uses default config + real model to measure actual values.
"""
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

import torch
import torch.nn.functional as F
from retlesionuni.config import load_config
from retlesionuni.models.retlesionuni import RetLesionUni
from retlesionuni.models.lpm import LesionPerceptionModule


def diagnose_lpm_loss_scale():
    """Problem B: Measure actual L1 loss between ViT attention maps."""
    print("=" * 60)
    print("DIAGNOSIS B: L_les scale")
    print("=" * 60)

    cfg = load_config(_project_root / "configs" / "default.yaml")
    model = RetLesionUni(cfg)
    model.eval()

    # Simulate two augmented views of same image
    img_size = cfg.model.img_size  # 512
    x1 = torch.randn(4, 3, img_size, img_size)  # view 1
    x2 = x1 + 0.02 * torch.randn_like(x1)       # view 2 (small perturbation)

    with torch.no_grad():
        enc1 = model.encoder(x1, return_attention=True)
        enc2 = model.encoder(x2, return_attention=True)

    attn1 = enc1["last_attention"]  # (B, num_heads, N+1, N+1)
    attn2 = enc2["last_attention"]

    # Process like LPM does
    attn1_avg = attn1.mean(dim=1)[:, 1:, 1:]  # (B, N, N)
    attn2_avg = attn2.mean(dim=1)[:, 1:, 1:]  # (B, N, N)

    N = attn1_avg.shape[-1]  # 1024

    # Raw L1 loss (what LPM computes)
    raw_l1 = F.l1_loss(attn1_avg, attn2_avg).item()
    print(f"  N patches: {N}")
    print(f"  attn element mean: {attn1_avg.mean().item():.6f} (theory 1/{N} = {1/N:.6f})")
    print(f"  attn element std:  {attn1_avg.std().item():.6f}")
    print(f"  raw L1(attn1, attn2): {raw_l1:.8f}")

    # With old lambda=0.1
    print(f"  L_les (lambda=0.1):  {raw_l1 * 0.1:.8f}")
    # With new lambda=10.0
    print(f"  L_les (lambda=10.0): {raw_l1 * 10.0:.8f}")
    # What lambda would give L_les ≈ 0.002 (0.3% of total loss ~0.7)?
    needed = 0.002 / max(raw_l1, 1e-10)
    print(f"  Lambda needed for L_les=0.002: {needed:.0f}")

    # Also check: is attention uniform or structured?
    # Uniform attention: all elements ≈ 1/N, std ≈ 0
    # Structured attention: some patches attend more, std > 0
    row_entropy = -(attn1_avg * torch.log(attn1_avg + 1e-12)).sum(dim=-1).mean()
    max_entropy = torch.log(torch.tensor(float(N)))
    print(f"  Attention entropy: {row_entropy.item():.4f} / max={max_entropy.item():.4f}")
    print(f"  => attention is {'structured' if row_entropy < 0.95 * max_entropy else 'nearly uniform'}")

    # Now test with LPM module directly
    lpm = LesionPerceptionModule()
    patches = enc1["patch_tokens"]
    enhanced, loss_raw = lpm(patches, attn1, attn2)
    print(f"  LPM.forward() raw loss: {loss_raw.item():.8f}")
    print(f"  With lambda=10.0: {loss_raw.item() * 10.0:.6f}")


def diagnose_ddr_head():
    """Problem A: Check DDR head design and output characteristics."""
    print()
    print("=" * 60)
    print("DIAGNOSIS A: DDR head check")
    print("=" * 60)

    cfg = load_config(_project_root / "configs" / "default.yaml")
    model = RetLesionUni(cfg)
    model.eval()

    img_size = cfg.model.img_size
    x = torch.randn(4, 3, img_size, img_size)

    with torch.no_grad():
        enc = model.encoder(x, return_multilevel=True)
        multilevel = enc["multilevel_features"]
        for i, t in enumerate(multilevel):
            layer = cfg.model.multilevel_layers[i]
            print(f"  Layer {layer}: tokens shape={list(t.shape)}, "
                  f"mean={t.mean().item():.4f}, std={t.std().item():.4f}, "
                  f"min={t.min().item():.4f}, max={t.max().item():.4f}")

        seg = model.ddr_head(multilevel, img_size)
        print(f"  DDR output: shape={list(seg.shape)}")
        print(f"    mean={seg.mean().item():.4f}, std={seg.std().item():.4f}")
        print(f"    min={seg.min().item():.4f}, max={seg.max().item():.4f}")
        print(f"    NaN? {torch.isnan(seg).any().item()}")
        print(f"    Inf? {torch.isinf(seg).any().item()}")

    # Check DDR head param count
    ddr_params = sum(p.numel() for p in model.ddr_head.parameters())
    print(f"  DDR head params: {ddr_params:,}")

    # Check: what does old checkpoint DDR head look like?
    old_ckpt_path = _project_root / "outputs" / "checkpoints" / "retlesionuni_full" / "resume_checkpoint_old_incompatible.pth"
    if old_ckpt_path.exists():
        old = torch.load(str(old_ckpt_path), map_location='cpu', weights_only=False)
        ddr_keys = [k for k in old["model_state_dict"].keys() if "ddr_head" in k]
        print(f"  Old DDR head keys ({len(ddr_keys)}):")
        for k in ddr_keys[:10]:
            print(f"    {k}: {list(old['model_state_dict'][k].shape)}")
        if len(ddr_keys) > 10:
            print(f"    ... and {len(ddr_keys) - 10} more")
        old_ddr_params = sum(
            old["model_state_dict"][k].numel() for k in ddr_keys
        )
        print(f"  Old DDR head params: {old_ddr_params:,}")
    else:
        print(f"  (no old checkpoint to compare)")


if __name__ == "__main__":
    diagnose_lpm_loss_scale()
    diagnose_ddr_head()
