"""RetLesionUni: Top-level model assembly."""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import ViTEncoder
from .heads import ODIRClassifier, DDRSegmentator
from .lpm import LesionPerceptionModule
from .ctam import CrossTaskAlignmentModule


class RetLesionUni(nn.Module):
    """Lesion-Aware Unified Foundation Model for Cross-Task Retinal Disease Analysis.

    Architecture:
        Shared ViT Encoder -> [LPM, CTAM, ODIR Head, DDR Head]

    Feature flow (fixed):
        cls_token(1024) + patch_pooled(1024) = 2048-dim base feature
        if CTAM: z_odir_enhanced(256) --proj--> 2048, ADD to base → 2048
        odir_head(2048) → 8-class prediction

    Args:
        config: Model configuration namespace.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        cfg = config.model

        # Shared encoder
        model_name = getattr(cfg, "backbone", "vit_large_patch16_224")
        use_ckpt = getattr(config.training, "gradient_checkpointing", False)
        self.encoder = ViTEncoder(
            model_name=model_name,
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            multilevel_layers=cfg.multilevel_layers,
            pretrained_path=cfg.pretrained_path if cfg.pretrained else None,
            freeze_ratio=0.0,
            use_gradient_checkpointing=use_ckpt,
        )

        # LPM
        self.lpm = LesionPerceptionModule() if cfg.lpm.enabled else None
        self.lambda_lesion = cfg.lpm.lambda_lesion

        # CTAM
        if cfg.ctam.enabled:
            self.ctam = CrossTaskAlignmentModule(
                dim_in=cfg.hidden_dim,
                dim_proj=cfg.ctam.proj_dim,
                num_heads=cfg.ctam.cross_attn_heads,
                alpha=cfg.ctam.residual_alpha,
                temperature=cfg.ctam.temperature,
            )
            # Project CTAM output (256) back to feature dim for additive fusion
            self.ctam_fusion = nn.Linear(cfg.ctam.proj_dim, cfg.hidden_dim * 2)
            # Learnable gate: small init (0.1) prevents CTAM from dominating,
            # and its absence during validation causes less distribution shift
            self.ctam_fusion_scale = nn.Parameter(torch.tensor(0.1))
        else:
            self.ctam = None
            self.ctam_fusion = None
        self.lambda_align = cfg.ctam.lambda_align

        # Task heads
        # ODIR input: cls_token(1024) + patch_pooled(1024) = 2048
        # CTAM features are fused additively (no dimension change)
        odir_dim_in = cfg.hidden_dim * 2  # 2048
        self.odir_head = ODIRClassifier(
            dim_in=odir_dim_in,
            num_classes=cfg.odir_num_classes,
        )
        self.ddr_head = DDRSegmentator(
            dim_in=cfg.hidden_dim,
            num_classes=cfg.ddr_num_classes,
            multilevel_layers=cfg.multilevel_layers,
        )

        # Flags for training stages
        self.use_lpm = False
        self.use_ctam = False
        self.warmup_factor = 1.0

    def set_stage_config(self, use_lpm: bool, use_ctam: bool, warmup_factor: float = 1.0):
        """Configure which modules and loss weights to use (for stage switching)."""
        self.use_lpm = use_lpm and self.lpm is not None
        self.use_ctam = use_ctam and self.ctam is not None
        self.warmup_factor = warmup_factor

    def forward(self, batch: dict) -> dict:
        """Forward pass with joint ODIR + DDR batch.

        Args:
            batch: Dict from JointDataLoader with keys:
                'odir': {'image_v1', 'image_v2', 'labels', 'dr_label'}
                'ddr':  {'image', 'mask'}

        Returns:
            Dict with keys:
                'odir_logits': (B_O, 8)
                'ddr_seg': (B_D, 5, H, W)
                'loss_lesion': scalar (0.0 if LPM disabled)
                'loss_align': scalar (0.0 if CTAM disabled)
        """
        odir = batch["odir"]
        ddr = batch["ddr"]

        x_odir_v1 = odir["image_v1"]
        x_odir_v2 = odir["image_v2"]
        x_ddr = ddr["image"]

        # --- ODIR path ---
        enc1 = self.encoder(x_odir_v1, return_attention=self.use_lpm)
        loss_lesion = torch.tensor(0.0, device=x_odir_v1.device)

        if self.use_lpm:
            enc2 = self.encoder(x_odir_v2, return_attention=True)
            f_odir_enhanced, loss_lesion = self.lpm(
                enc1["patch_tokens"],
                enc1["last_attention"],
                enc2["last_attention"],
            )
            # Fix 1: Pool LPM-enhanced patch tokens for classification
            f_odir_pooled = f_odir_enhanced.mean(dim=1)  # (B, 1024)
        else:
            # Fix 7: Pool raw patch tokens when LPM is off
            f_odir_pooled = enc1["patch_tokens"].mean(dim=1)  # (B, 1024)

        # Fix 7: Fuse CLS token (global semantics) with patch-pooled (local details)
        f_odir_for_cls = torch.cat([enc1["cls_token"], f_odir_pooled], dim=-1)  # (B, 2048)

        # --- DDR path ---
        enc_ddr = self.encoder(x_ddr, return_multilevel=True)
        ddr_seg = self.ddr_head(enc_ddr["multilevel_features"], self.config.model.img_size)

        # --- CTAM ---
        loss_align = torch.tensor(0.0, device=x_odir_v1.device)
        if self.use_ctam:
            dr_labels = odir["dr_label"].to(x_odir_v1.device)
            z_odir_enhanced, z_ddr_aligned, loss_align = self.ctam(
                enc1["cls_token"],
                enc_ddr["cls_token"],
                dr_labels,
            )
            # Fix 2: Additive fusion of CTAM features (256→2048 projection, then add)
            # warmup_factor gates both loss_align and ctam_fused to prevent early noise
            ctam_fused = self.ctam_fusion(z_odir_enhanced)  # (B, 256) → (B, 2048)
            f_odir_for_cls = f_odir_for_cls + self.ctam_fusion_scale * self.warmup_factor * ctam_fused

        odir_logits = self.odir_head(f_odir_for_cls)

        return {
            "odir_logits": odir_logits,
            "ddr_seg": ddr_seg,
            "loss_lesion": loss_lesion * self.lambda_lesion * self.warmup_factor,
            "loss_align": loss_align * self.lambda_align * self.warmup_factor,
        }
