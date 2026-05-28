"""ODIR Dataset: multi-label classification with dual augmentation views."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .preprocess_odir import ODIR_CLASS_NAMES


class ODIRDataset(Dataset):
    """ODIR multi-label classification dataset.

    Each __getitem__ returns two augmented views of the same image (for LPM).

    Returns: {
        'image_v1': (3, H, W) tensor,
        'image_v2': (3, H, W) tensor,
        'labels': (8,) float tensor,
        'dr_label': int (0 or 1),
        'image_id': str,
    }
    """

    def __init__(
        self,
        cache_dir: str | Path,
        split: str,
        transform_v1=None,
        transform_v2=None,
    ):
        self.cache_dir = Path(cache_dir) / split
        self.split = split
        self.transform_v1 = transform_v1
        self.transform_v2 = transform_v2

        metadata_path = self.cache_dir / "metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found: {metadata_path}. Run preprocessing first."
            )
        self.df = pd.read_csv(metadata_path)

        # Build DR-positive index for weighted sampling
        self.dr_positive_indices = self.df[self.df["dr_label"] == 1].index.tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image_id = row["image_id"]

        img_path = row["image_path"]
        if not Path(img_path).exists():
            img_path = self.cache_dir / "images" / f"{image_id}.jpg"
        image = np.array(Image.open(img_path).convert("RGB"))

        labels = torch.tensor(
            [int(row[name]) for name in ODIR_CLASS_NAMES], dtype=torch.float32
        )
        dr_label = int(row["dr_label"])

        # Apply transforms (two different augmentations)
        if self.transform_v1 is not None:
            aug1 = self.transform_v1(image=image)["image"]
        else:
            aug1 = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        if self.transform_v2 is not None:
            aug2 = self.transform_v2(image=image)["image"]
        else:
            aug2 = aug1.clone()

        return {
            "image_v1": aug1,
            "image_v2": aug2,
            "labels": labels,
            "dr_label": dr_label,
            "image_id": image_id,
        }

    def get_sample_weights(self, dr_positive_weight: float = 3.0) -> torch.Tensor:
        """Get per-sample weights for WeightedRandomSampler.

        Args:
            dr_positive_weight: Weight multiplier for DR-positive samples.

        Returns:
            Tensor of shape (len(dataset),) with per-sample weights.
        """
        weights = torch.ones(len(self), dtype=torch.float32)
        for idx in self.dr_positive_indices:
            weights[idx] = dr_positive_weight
        return weights
