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

    Args:
        config: Model configuration namespace.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        cfg = config.model

        # Shared encoder
        model_name = getattr(cfg, "backbone", "vit_large_patch16_224")
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
        else:
            self.ctam = None
        self.lambda_align = cfg.ctam.lambda_align

        # Task heads
        self.odir_head = ODIRClassifier(
            dim_in=cfg.hidden_dim,
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
        else:
            f_odir_enhanced = enc1["patch_tokens"]

        odir_logits = self.odir_head(enc1["cls_token"])

        # --- DDR path ---
        enc_ddr = self.encoder(x_ddr, return_multilevel=True)
        ddr_seg = self.ddr_head(enc_ddr["multilevel_features"], self.config.model.img_size)

        # --- CTAM ---
        loss_align = torch.tensor(0.0, device=x_odir_v1.device)
        if self.use_ctam:
            dr_labels = odir["dr_label"].to(x_odir_v1.device)
            _, _, loss_align = self.ctam(
                enc1["cls_token"],
                enc_ddr["cls_token"],
                dr_labels,
            )

        return {
            "odir_logits": odir_logits,
            "ddr_seg": ddr_seg,
            "loss_lesion": loss_lesion * self.lambda_lesion * self.warmup_factor,
            "loss_align": loss_align * self.lambda_align * self.warmup_factor,
        }
