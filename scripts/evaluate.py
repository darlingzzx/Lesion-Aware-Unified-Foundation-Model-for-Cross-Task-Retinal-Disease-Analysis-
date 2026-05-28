#!/usr/bin/env python
"""Evaluation entry point for RetLesionUni.

Usage:
    python scripts/evaluate.py --checkpoint outputs/checkpoints/retlesionuni_full/best_model.pth
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

from retlesionuni.config import load_config
from retlesionuni.data.transforms import get_ddr_transform, get_odir_eval_transform
from retlesionuni.data.joint_loader import DataLoaderFactory
from retlesionuni.eval.metrics import compute_ddr_metrics, compute_odir_metrics
from retlesionuni.models.retlesionuni import RetLesionUni


def main():
    parser = argparse.ArgumentParser(description="RetLesionUni Evaluation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")

    # Load model
    model = RetLesionUni(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.set_stage_config(use_lpm=False, use_ctam=False)

    # Setup data
    factory = DataLoaderFactory(config)
    odir_eval_tr = get_odir_eval_transform(config.model.img_size)
    ddr_eval_tr = get_ddr_transform(config.model.img_size, is_train=False)

    _, _, odir_test_loader = factory.create_odir_loaders(None, None, odir_eval_tr)
    _, _, ddr_test_loader = factory.create_ddr_loaders(None, ddr_eval_tr)

    # --- ODIR Evaluation ---
    print("\nEvaluating ODIR...")
    all_odir_preds, all_odir_targets = [], []
    with torch.no_grad():
        for batch in tqdm(odir_test_loader, desc="ODIR"):
            x = batch["image_v1"].to(device)
            enc = model.encoder(x)
            preds = model.odir_head(enc["cls_token"]).cpu()
            all_odir_preds.append(preds)
            all_odir_targets.append(batch["labels"])
    odir_preds = torch.cat(all_odir_preds).numpy()
    odir_targets = torch.cat(all_odir_targets).numpy()
    odir_metrics = compute_odir_metrics(odir_preds, odir_targets)
    print(f"  ODIR: Acc={odir_metrics['accuracy']:.4f}, F1={odir_metrics['f1']:.4f}, AUC={odir_metrics['auc']:.4f}")

    # --- DDR Evaluation ---
    print("\nEvaluating DDR...")
    all_ddr_preds, all_ddr_targets = [], []
    with torch.no_grad():
        for batch in tqdm(ddr_test_loader, desc="DDR"):
            x = batch["image"].to(device)
            enc = model.encoder(x, return_multilevel=True)
            preds = model.ddr_head(enc["multilevel_features"], config.model.img_size).cpu()
            all_ddr_preds.append(preds)
            all_ddr_targets.append(batch["mask"])
    ddr_preds = torch.cat(all_ddr_preds).numpy()
    ddr_targets = torch.cat(all_ddr_targets).numpy()
    ddr_metrics = compute_ddr_metrics(ddr_preds, ddr_targets)
    print(f"  DDR: mIoU={ddr_metrics['mIoU']:.4f}, mDice={ddr_metrics['mDice']:.4f}")
    print(f"  Lesion-only: mIoU={ddr_metrics['lesion_mIoU']:.4f}, mDice={ddr_metrics['lesion_mDice']:.4f}")

    # Print per-class results
    class_names = ["BG", "EX", "HE", "MA", "SE"]
    print("\nPer-class Dice:")
    for i, name in enumerate(class_names):
        print(f"  {name}: Dice={ddr_metrics['per_class_dice'][i]:.4f}, IoU={ddr_metrics['per_class_iou'][i]:.4f}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
