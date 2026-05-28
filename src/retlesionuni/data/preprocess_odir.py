"""ODIR dataset preprocessing: patient-level -> image-level labels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm


# ODIR 8-class label columns
ODIR_LABEL_COLS = ["N", "D", "G", "C", "A", "H", "M", "O"]
ODIR_CLASS_NAMES = [
    "Normal", "Diabetes", "Glaucoma", "Cataract",
    "AMD", "Hypertension", "Myopia", "Other",
]


def preprocess_odir_split(
    data_dir: Path,
    output_dir: Path,
    split_name: str,
) -> Path:
    """Preprocess one ODIR split (train/valid/test).

    Expands patient-level annotations to image-level: each patient with two eyes
    produces two image records, both sharing the patient's multi-label vector.

    Args:
        data_dir: Path to ODIR/{split}/ (contains JPG images and {split}_annotations.xlsx).
        output_dir: Path to save preprocessed outputs.
        split_name: 'train', 'valid', or 'test'.

    Returns:
        Path to the generated metadata CSV file.
    """
    (output_dir / "images").mkdir(parents=True, exist_ok=True)

    xlsx_path = data_dir / f"{split_name}_annotations.xlsx"
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {xlsx_path}")

    df = pd.read_excel(xlsx_path)
    records: list = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"ODIR {split_name}"):
        labels = {col: int(row.get(col, 0)) for col in ODIR_LABEL_COLS}
        dr_label = labels.get("D", 0)

        for eye in ["left", "right"]:
            fundus_col = f"{'Left' if eye == 'left' else 'Right'}-Fundus"
            img_filename = row.get(fundus_col, "")

            if not img_filename or pd.isna(img_filename):
                continue

            img_path = data_dir / str(img_filename)
            if not img_path.exists():
                # Try case-insensitive
                candidates = list(data_dir.glob(f"*{img_filename[-10:]}"))
                if candidates:
                    img_path = candidates[0]
                else:
                    continue

            records.append({
                "image_id": Path(str(img_filename)).stem,
                "image_path": str(img_path),
                "patient_id": int(row.get("ID", 0)),
                "eye": eye,
                **{ODIR_CLASS_NAMES[i]: labels[col] for i, col in enumerate(ODIR_LABEL_COLS)},
                "dr_label": dr_label,
            })

    metadata_path = output_dir / "metadata.csv"
    pd.DataFrame(records).to_csv(metadata_path, index=False)
    return metadata_path
