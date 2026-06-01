"""
Master script: Download RetFound weights -> Train -> Evaluate
Run this once and it handles everything automatically.
"""
import subprocess
import sys
import time
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = "C:/Users/zhangjuntao/.conda/envs/dlenv/python.exe"
WEIGHTS_PATH = PROJECT_ROOT / "pretrained" / "RETFound_cfp_weights.pth"
ENV = os.environ.copy()
ENV["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_step(step_name, args, timeout=None):
    print(f"\n{'='*70}")
    print(f"STEP: {step_name}")
    print(f"{'='*70}")
    cmd = [PYTHON] + args
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=ENV, timeout=timeout)
    if result.returncode != 0:
        print(f"\nERROR: {step_name} failed with code {result.returncode}")
        return False
    print(f"\nDONE: {step_name}")
    return True

def check_weights():
    if WEIGHTS_PATH.exists():
        sz = WEIGHTS_PATH.stat().st_size
        if sz > 3.5 * 1024**3:
            return True
        else:
            print(f"Removing incomplete weights ({sz/1024/1024:.1f} MB)")
            WEIGHTS_PATH.unlink()
    return False

def main():
    print("RetLesionUni Full Pipeline")
    print(f"Project: {PROJECT_ROOT}")
    print(f"Python: {PYTHON}")
    print(f"GPU:", end=" ")
    subprocess.run([PYTHON, "-c", "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND')"])

    # Step 1: Download weights
    if not check_weights():
        print("\nDownloading RetFound weights (this may take a while)...")
        download_script = PROJECT_ROOT / "scripts" / "download_retfound.py"
        if not run_step("Download RetFound Weights", [str(download_script)], timeout=None):
            print("\nFATAL: Could not download RetFound weights")
            print(f"Please manually download from Google Drive and place at: {WEIGHTS_PATH}")
            print("URL: https://drive.google.com/uc?id=1l62zbWUFTlp214SvK6eMwPQZAzcwoeBE")
            sys.exit(1)
    else:
        print(f"\nRetFound weights already present: {WEIGHTS_PATH.stat().st_size/1024**3:.2f} GB")

    # Step 2: Verify preprocessed data
    print("\nVerifying preprocessed data...")
    result = subprocess.run(
        [PYTHON, "-c", """
import pandas as pd
from pathlib import Path
cache = Path('outputs/preprocessed')
for ds in ['ddr', 'odir']:
    for split in ['train', 'valid', 'test']:
        df = pd.read_csv(cache / ds / split / 'metadata.csv')
        print(f'  {ds}/{split}: {len(df)} samples OK')
"""],
        cwd=str(PROJECT_ROOT), env=ENV, capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR: Preprocessed data incomplete, run: python scripts/preprocess.py")
        sys.exit(1)

    # Step 3: Run quick tests
    if not run_step("Run Tests (model forward)", ["tests/test_model_forward.py"], timeout=300):
        print("WARNING: Model tests failed, but continuing...")

    # Step 4: Start training
    train_args = ["scripts/train.py", "--config", "configs/default.yaml"]
    if not run_step("Training", train_args, timeout=None):
        print("\nERROR: Training failed!")
        sys.exit(1)

    # Step 5: Evaluate
    checkpoints_dir = PROJECT_ROOT / "outputs" / "checkpoints" / "retlesionuni_full"
    best_model = checkpoints_dir / "best_model.pth"
    if best_model.exists():
        eval_args = ["scripts/evaluate.py", "--config", "configs/default.yaml", "--checkpoint", str(best_model)]
        run_step("Evaluate Best Model", eval_args, timeout=3600)
    else:
        # Try final model
        final_model = checkpoints_dir / "final_model.pth"
        if final_model.exists():
            eval_args = ["scripts/evaluate.py", "--config", "configs/default.yaml", "--checkpoint", str(final_model)]
            run_step("Evaluate Final Model", eval_args, timeout=3600)
        else:
            print("No checkpoint found for evaluation")

    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()
