"""Data augmentation and preprocessing transforms using albumentations."""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_odir_transforms(img_size: int) -> tuple[A.Compose, A.Compose]:
    """Get ODIR transform pipelines (two augmentation views).

    Returns:
        (transform_v1, transform_v2): Two similar but distinct augmentation pipelines.
    """
    base = _odir_base(img_size)
    aug1 = _odir_augment()
    aug2 = _odir_augment()

    transform_v1 = A.Compose(list(base) + aug1 + [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])
    transform_v2 = A.Compose(list(base) + aug2 + [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])

    return transform_v1, transform_v2


def get_odir_eval_transform(img_size: int) -> A.Compose:
    """Get ODIR evaluation transform (no augmentation, just resize + normalize)."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_ddr_transform(img_size: int, is_train: bool = True) -> A.Compose:
    """Get DDR transforms (image + mask)."""
    transforms = [A.Resize(img_size, img_size)]
    if is_train:
        transforms += [
            A.Rotate(limit=30, border_mode=0, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ]
    transforms += [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    return A.Compose(transforms)


def _odir_base(img_size: int) -> list:
    return [A.Resize(img_size, img_size)]


def _odir_augment() -> list:
    return [
        A.Rotate(limit=30, border_mode=0, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 32), hole_width_range=(8, 32), p=0.5),
    ]
