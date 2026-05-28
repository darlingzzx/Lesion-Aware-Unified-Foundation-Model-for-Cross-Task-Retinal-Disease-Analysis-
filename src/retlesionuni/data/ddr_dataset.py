"""DDR Dataset: image + 5-class segmentation mask."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class DDRDataset(Dataset):
    """DDR segmentation dataset.

    Returns: {
        'image': (3, H, W) tensor,
        'mask': (H, W) long tensor (0=bg, 1=EX, 2=HE, 3=MA, 4=SE),
        'image_id': str,
    }
    """

    def __init__(self, cache_dir: str | Path, split: str, transform=None):
        self.cache_dir = Path(cache_dir) / split
        self.split = split
        self.transform = transform

        metadata_path = self.cache_dir / "metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found: {metadata_path}. Run preprocessing first."
            )
        self.df = pd.read_csv(metadata_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image_id = row["image_id"]

        # Load image
        img_path = row["image_path"]
        if not Path(img_path).exists():
            # Try the preprocessed images directory
            img_path = self.cache_dir / "images" / f"{image_id}.jpg"
        image = np.array(Image.open(img_path).convert("RGB"))

        # Load mask
        mask_path = self.cache_dir / "masks" / f"{image_id}.npy"
        mask = np.load(mask_path).astype(np.int64)

        # Apply transforms (image + mask together)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return {
            "image": image,
            "mask": mask,
            "image_id": image_id,
        }

    def get_bboxes(self, image_id: str) -> list:
        """Load bounding boxes for an image (for explainability analysis)."""
        bbox_path = self.cache_dir / "bboxes" / f"{image_id}_bboxes.json"
        if bbox_path.exists():
            with open(bbox_path) as f:
                return json.load(f)
        return []
