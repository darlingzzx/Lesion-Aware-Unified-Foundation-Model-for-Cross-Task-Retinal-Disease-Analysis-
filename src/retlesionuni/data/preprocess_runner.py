"""One-shot data preprocessing runner for RetLesionUni."""

from __future__ import annotations

from pathlib import Path

from .preprocess_ddr import preprocess_ddr_split
from .preprocess_odir import preprocess_odir_split


def run_preprocessing(data_cfg, preprocessed_cache: str | Path) -> None:
    """Run all preprocessing steps: DDR + ODIR for all splits.

    Args:
        data_cfg: Data config namespace (from config YAML).
        preprocessed_cache: Root directory for preprocessed cache.
    """
    cache_dir = Path(preprocessed_cache)
    splits = data_cfg.splits  # ["train", "valid", "test"]

    ddr_data_root = Path(data_cfg.ddr_root)
    odir_data_root = Path(data_cfg.odir_root)

    for split in splits:
        print(f"\n{'='*60}")
        print(f"Processing split: {split}")
        print(f"{'='*60}")

        # --- DDR ---
        detection_dir = ddr_data_root / data_cfg.ddr_detection_dir / split
        segmentation_dir = ddr_data_root / data_cfg.ddr_segmentation_dir / split
        ddr_out = cache_dir / "ddr" / split

        if detection_dir.is_dir() and segmentation_dir.is_dir():
            print(f"\n[DDR {split}] Detection: {detection_dir}")
            print(f"[DDR {split}] Segmentation: {segmentation_dir}")
            metadata_path = preprocess_ddr_split(
                detection_dir=detection_dir,
                segmentation_dir=segmentation_dir,
                output_dir=ddr_out,
                split_name=split,
            )
            print(f"[DDR {split}] Metadata saved to: {metadata_path}")
        else:
            print(f"[DDR {split}] SKIP — directories not found.")

        # --- ODIR ---
        odir_split_dir = odir_data_root / split
        odir_out = cache_dir / "odir" / split

        if odir_split_dir.is_dir():
            print(f"\n[ODIR {split}] Data dir: {odir_split_dir}")
            metadata_path = preprocess_odir_split(
                data_dir=odir_split_dir,
                output_dir=odir_out,
                split_name=split,
            )
            print(f"[ODIR {split}] Metadata saved to: {metadata_path}")
        else:
            print(f"[ODIR {split}] SKIP — directory not found.")

    print(f"\n{'='*60}")
    print("Preprocessing complete!")
    print(f"Cache located at: {cache_dir}")
