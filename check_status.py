"""Check RetLesionUni project status — run anytime to see what's ready."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "pretrained" / "RETFound_cfp_weights.pth"

print("=" * 50)
print("RetLesionUni Status Check")
print("=" * 50)

# 1. Weights
print("\n[1] RetFound Weights:")
if WEIGHTS.exists():
    gb = WEIGHTS.stat().st_size / 1024**3
    if gb > 3.5:
        print(f"  DONE: {gb:.2f} GB — ready to train!")
    else:
        print(f"  PARTIAL: {gb:.2f} GB — download incomplete, delete it and retry")
else:
    part_files = list(ROOT.glob("pretrained/*.part"))
    if part_files:
        for pf in part_files:
            mb = pf.stat().st_size / 1024**2
            print(f"  DOWNLOADING: {pf.name} ({mb:.0f} MB so far)")
    else:
        print("  NOT STARTED: run scripts/download_retfound.py")

# 2. Preprocessed data
print("\n[2] Preprocessed Data:")
cache = ROOT / "outputs" / "preprocessed"
for ds in ["ddr", "odir"]:
    for split in ["train", "valid", "test"]:
        meta = cache / ds / split / "metadata.csv"
        if meta.exists():
            import pandas as pd
            n = len(pd.read_csv(meta))
            print(f"  {ds}/{split}: {n} samples")
        else:
            print(f"  {ds}/{split}: MISSING!")

# 3. GPU
print("\n[3] GPU:")
try:
    import torch
    if torch.cuda.is_available():
        print(f"  {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem/1024**3:.0f} GB)")
    else:
        print("  NOT AVAILABLE")
except:
    print("  Could not check (run with dlenv python)")

# 4. Dependencies
print("\n[4] Key Packages:")
for pkg in ["torch", "timm", "transformers", "albumentations", "monai", "cv2"]:
    try:
        if pkg == "cv2":
            import cv2
            print(f"  opencv-python: {cv2.__version__}")
        else:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            print(f"  {pkg}: {ver}")
    except:
        print(f"  {pkg}: NOT INSTALLED")

# 5. Ready-to-run command
print("\n[5] Next Step:")
if WEIGHTS.exists() and WEIGHTS.stat().st_size > 3.5 * 1024**3:
    print("  READY! Run: python scripts/train.py")
    print("  Or full pipeline: python scripts/run_all.py")
else:
    print("  Weights not ready yet.")
    print("  Download script running in background (check pretrained/ for .part files)")
    print("  OR manually download from:")
    print("  https://drive.google.com/uc?id=1l62zbWUFTlp214SvK6eMwPQZAzcwoeBE")
    print(f"  Save to: {WEIGHTS}")

print("=" * 50)
