"""DDR dataset preprocessing: XML deduplication + TIFF mask assembly."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
from tqdm import tqdm


# Class mapping: XML lowercase names -> integer labels
LESION_CLASS_MAP: Dict[str, int] = {
    "ex": 1,   # Hard exudates
    "he": 2,   # Hemorrhages
    "ma": 3,   # Microaneurysms
    "se": 4,   # Soft exudates
}

# TIFF mask directory name: (upper) class name -> integer label
LESION_TIFF_MAP: Dict[str, int] = {
    "EX": 1,
    "HE": 2,
    "MA": 3,
    "SE": 4,
}


def parse_xml_bboxes(xml_path: str | Path) -> List[dict]:
    """Parse a Pascal VOC XML file, deduplicate bboxes, return unique annotations.

    Returns:
        List of dicts: [{'class': 'ma', 'bbox': [xmin, ymin, xmax, ymax]}, ...]
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    seen: set = set()
    unique_objects: list = []

    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower() if obj.find("name").text else ""
        bbox_el = obj.find("bndbox")
        xmin = int(float(bbox_el.find("xmin").text))
        ymin = int(float(bbox_el.find("ymin").text))
        xmax = int(float(bbox_el.find("xmax").text))
        ymax = int(float(bbox_el.find("ymax").text))

        dedup_key = (name, xmin, ymin, xmax, ymax)
        if dedup_key not in seen:
            seen.add(dedup_key)
            unique_objects.append({
                "class": name,
                "bbox": [xmin, ymin, xmax, ymax],
            })

    return unique_objects


def build_segmentation_mask(image_id: str, mask_dirs: Dict[str, Path]) -> Tuple[np.ndarray, bool]:
    """Assemble a 5-class segmentation mask from 4 per-lesion TIFF files.

    Args:
        image_id: Image filename stem (e.g., '007-1774-100').
        mask_dirs: Dict mapping lesion class (uppercase) to directory Path.

    Returns:
        (mask, success): mask is (H, W) uint8 array with values 0-4, or None if any
        required mask file is missing.
    """
    # Load the first TIFF to get image dimensions
    first_cls = list(mask_dirs.keys())[0]
    first_path = mask_dirs[first_cls] / f"{image_id}.tif"

    if not first_path.exists():
        return None, False

    ref_mask = cv2.imread(str(first_path), cv2.IMREAD_GRAYSCALE)
    if ref_mask is None:
        return None, False
    h, w = ref_mask.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    for cls_name, cls_dir in mask_dirs.items():
        label_val = LESION_TIFF_MAP.get(cls_name.upper(), 0)
        if label_val == 0:
            continue

        tif_path = cls_dir / f"{image_id}.tif"
        if not tif_path.exists():
            continue

        cls_mask = cv2.imread(str(tif_path), cv2.IMREAD_GRAYSCALE)
        if cls_mask is None:
            continue

        # Normalize: 0 remains 0, nonzero -> 1
        cls_binary = (cls_mask > 0).astype(np.uint8)
        mask[cls_binary > 0] = label_val

    return mask, True


def detect_label_directory(seg_dir: Path) -> str:
    """Auto-detect the label directory name for a segmentation split.

    DDR uses 'label' for train/test but 'segmentation label' for valid.
    """
    candidates = ["label", "segmentation label"]
    for c in candidates:
        full = seg_dir / c
        if full.is_dir():
            return c
    raise FileNotFoundError(f"No label directory found in {seg_dir}. Candidates: {candidates}")


def preprocess_ddr_split(
    detection_dir: Path,
    segmentation_dir: Path,
    output_dir: Path,
    split_name: str,
) -> Path:
    """Preprocess one DDR split (train/valid/test).

    Args:
        detection_dir: Path to lesion_detection/{split}/ (contains .xml files).
        segmentation_dir: Path to lesion_segmentation/{split}/ (contains image/ and label/).
        output_dir: Path to save preprocessed outputs.
        split_name: 'train', 'valid', or 'test'.

    Returns:
        Path to the generated metadata CSV file.
    """
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)
    (output_dir / "bboxes").mkdir(parents=True, exist_ok=True)

    image_dir = segmentation_dir / "image"
    label_dir_name = detect_label_directory(segmentation_dir)
    label_base = segmentation_dir / label_dir_name

    image_files = sorted(image_dir.glob("*.jpg"))
    records: list = []

    for img_path in tqdm(image_files, desc=f"DDR {split_name}"):
        stem = img_path.stem  # e.g., "007-1774-100"

        # --- Bounding boxes (detection) ---
        xml_path = detection_dir / f"{stem}.xml"
        bboxes = []
        if xml_path.exists():
            bboxes = parse_xml_bboxes(xml_path)

        bbox_out = output_dir / "bboxes" / f"{stem}_bboxes.json"
        with open(bbox_out, "w") as f:
            json.dump(bboxes, f)

        # --- Segmentation mask ---
        mask_dirs = {
            cls_name: label_base / cls_name
            for cls_name in LESION_TIFF_MAP
        }
        mask, success = build_segmentation_mask(stem, mask_dirs)

        if success and mask is not None:
            mask_out = output_dir / "masks" / f"{stem}.npy"
            np.save(mask_out, mask)

            # Per-lesion presence flags
            has_ex = int((mask == 1).any())
            has_he = int((mask == 2).any())
            has_ma = int((mask == 3).any())
            has_se = int((mask == 4).any())
        else:
            has_ex = has_he = has_ma = has_se = 0

        records.append({
            "image_id": stem,
            "image_path": str(img_path),
            "has_ex": has_ex,
            "has_he": has_he,
            "has_ma": has_ma,
            "has_se": has_se,
            "num_bboxes": len(bboxes),
        })

    # Save metadata
    import pandas as pd
    metadata_path = output_dir / f"metadata.csv"
    pd.DataFrame(records).to_csv(metadata_path, index=False)
    return metadata_path
