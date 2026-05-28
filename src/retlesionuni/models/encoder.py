"""RetFound / ViT-Large encoder with position embedding interpolation and multi-level feature extraction."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTEncoder(nn.Module):
    """ViT encoder that supports position embedding interpolation, frozen layers,
    multi-level hidden states, and attention map extraction.

    If pretrained_path is provided, loads RetFound weights.
    Otherwise uses random initialization (e.g., timm pretrained or scratch).

    Args:
        img_size: Input image size (default 512).
        patch_size: Patch size (default 16).
        hidden_dim: Hidden dimension (default 1024 for ViT-Large).
        num_layers: Number of transformer layers (default 24 for ViT-Large).
        num_heads: Number of attention heads (default 16).
        mlp_ratio: MLP hidden ratio (default 4).
        multilevel_layers: Which layer indices to extract features from (1-indexed).
        pretrained_path: Path to RetFound .pth or None for random init.
        freeze_ratio: Fraction of layers to freeze (0.0 = none, 0.7 = first 70%).
    """

    def __init__(
        self,
        model_name: str = "vit_large_patch16_224",
        img_size: int = 512,
        patch_size: int = 16,
        hidden_dim: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        multilevel_layers: Optional[List[int]] = None,
        pretrained_path: Optional[str] = None,
        freeze_ratio: float = 0.0,
    ):
        super().__init__()
        self.model_name = model_name
        self.img_size = img_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_patches = (img_size // patch_size) ** 2
        self.multilevel_layers = multilevel_layers or [12, 18, 24]

        # Use timm to create the ViT model
        import timm

        self.vit = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            img_size=img_size,
        )

        # Interpolate position embedding if the model's default size differs
        if hasattr(self.vit, "pos_embed"):
            default_grid = int((self.vit.pos_embed.shape[1] - 1) ** 0.5)
            actual_grid = img_size // patch_size
            if default_grid != actual_grid:
                self._interpolate_pos_embed(default_grid, actual_grid)

        # Load pretrained weights if provided
        if pretrained_path and pretrained_path not in ("true", "false", ""):
            self._load_retfound_weights(pretrained_path)

        # Apply freezing
        if freeze_ratio > 0:
            self.freeze_layers(freeze_ratio)

    def _interpolate_pos_embed(self, old_grid_size: int, new_grid_size: int):
        """Interpolate position embeddings from old grid to new grid."""
        if old_grid_size == new_grid_size:
            return

        pos_embed = self.vit.pos_embed  # (1, N+1, D)
        cls_token = pos_embed[:, :1, :]
        pos_tokens = pos_embed[:, 1:, :]  # (1, N, D)

        D = pos_tokens.shape[-1]
        pos_tokens = pos_tokens.reshape(1, old_grid_size, old_grid_size, D).permute(0, 3, 1, 2)
        pos_tokens = F.interpolate(
            pos_tokens, size=(new_grid_size, new_grid_size), mode="bilinear", align_corners=False
        )
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, new_grid_size * new_grid_size, D)

        self.vit.pos_embed = nn.Parameter(torch.cat([cls_token, pos_tokens], dim=1))

    def _load_retfound_weights(self, pretrained_path: str):
        """Load RetFound weights from the official checkpoint format.

        The official RetFound checkpoint has:
          checkpoint['model'] -> state_dict with MAE ViT keys
          Keys to remove: head.weight, head.bias, fc_norm.weight, fc_norm.bias
        """
        import os
        if not os.path.exists(pretrained_path):
            print(f"[Encoder] WARNING: Pretrained weights not found at {pretrained_path}")
            print("[Encoder] Using random initialization.")
            return

        checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=False)

        # RetFound format: checkpoint has 'model' key containing state_dict
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Remove classification head keys (we use our own heads)
        head_keys = ["head.weight", "head.bias", "fc_norm.weight", "fc_norm.bias"]
        for k in head_keys:
            if k in state_dict:
                del state_dict[k]

        # Try to load with strict=False to handle remaining mismatches
        missing, unexpected = self.vit.load_state_dict(state_dict, strict=False)

        # Filter to only meaningful missing keys
        important_missing = [k for k in missing
                            if not any(hk in k for hk in ["head.", "fc_norm."])]
        if important_missing:
            print(f"[Encoder] Missing keys ({len(important_missing)}), first 5: {important_missing[:5]}")
        if unexpected:
            print(f"[Encoder] Unexpected keys ({len(unexpected)}), first 5: {unexpected[:5]}")

        print(f"[Encoder] Loaded RetFound weights from {pretrained_path}")

    def freeze_layers(self, freeze_ratio: float):
        """Freeze the first `freeze_ratio` fraction of transformer blocks."""
        blocks = self.vit.blocks
        num_freeze = int(len(blocks) * freeze_ratio)
        for i in range(num_freeze):
            for param in blocks[i].parameters():
                param.requires_grad = False
        print(f"[Encoder] Frozen {num_freeze}/{len(blocks)} layers ({freeze_ratio:.0%})")

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.vit.parameters():
            param.requires_grad = True

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        return_multilevel: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: (B, 3, H, W) input images.
            return_attention: If True, also return last layer attention maps.
            return_multilevel: If True, also return hidden states at multilevel_layers.

        Returns:
            Dict with keys:
                'cls_token': (B, hidden_dim)
                'patch_tokens': (B, num_patches, hidden_dim)
                'multilevel_features': List of (B, num_patches, hidden_dim) if return_multilevel
                'last_attention': (B, num_heads, num_patches+1, num_patches+1) if return_attention
        """
        B = x.shape[0]

        # Patch embedding
        x = self.vit.patch_embed(x)  # (B, num_patches, hidden_dim)

        # Add CLS token
        cls_token = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # (B, num_patches+1, hidden_dim)

        # Add position embedding
        x = x + self.vit.pos_embed

        # Apply dropout
        x = self.vit.pos_drop(x)

        # Pass through transformer blocks
        multilevel_features: list = []
        last_attention: Optional[torch.Tensor] = None

        for i, block in enumerate(self.vit.blocks, start=1):
            if return_attention and i == len(self.vit.blocks):
                # For last layer, we need attention maps
                attn_out, attn_weights = self._block_forward_with_attention(block, x)
                x = attn_out
                last_attention = attn_weights
            else:
                x = block(x)

            if return_multilevel and i in self.multilevel_layers:
                multilevel_features.append(x[:, 1:, :])  # Remove CLS token

        # Apply norm
        x = self.vit.norm(x)

        result = {
            "cls_token": x[:, 0, :],
            "patch_tokens": x[:, 1:, :],
        }

        if return_multilevel:
            result["multilevel_features"] = multilevel_features

        if return_attention and last_attention is not None:
            result["last_attention"] = last_attention

        return result

    def _block_forward_with_attention(self, block, x):
        """Forward through a single block, returning both output and attention weights."""
        # Save original forward to get attention
        y = block.norm1(x)
        qkv = block.attn.qkv(y).reshape(
            x.shape[0], x.shape[1], 3, block.attn.num_heads, -1
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_weights = (q @ k.transpose(-2, -1)) * block.attn.scale
        attn_weights = attn_weights.softmax(dim=-1)
        attn_out = (attn_weights @ v).transpose(1, 2).reshape(
            x.shape[0], x.shape[1], -1
        )
        attn_out = block.attn.proj(attn_out)
        x = x + block.drop_path1(block.ls1(attn_out))
        x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
        return x, attn_weights
