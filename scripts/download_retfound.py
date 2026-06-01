"""
Simple gdown retry loop. Each attempt downloads fresh; on failure, retries.
Google Drive typically drops after 15-20 min, so this needs many retries.
"""
import sys, time, os
import gdown
from pathlib import Path

URL = "https://drive.google.com/uc?id=1l62zbWUFTlp214SvK6eMwPQZAzcwoeBE"
OUTPUT = Path("D:/RetLesionUni/Lesion-Aware-Unified-Foundation-Model-for-Cross-Task-Retinal-Disease-Analysis-/pretrained/RETFound_cfp_weights.pth")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

MAX_RETRIES = 1000
for attempt in range(1, MAX_RETRIES + 1):
    # Clean up any partial files
    for f in OUTPUT.parent.glob("*"):
        if f.name != OUTPUT.name and (f.suffix in ('.part', '.tmp') or '.part' in f.name):
            try: f.unlink()
            except: pass

    if OUTPUT.exists():
        sz = OUTPUT.stat().st_size
        if sz > 3.5 * 1024**3:
            print(f"SUCCESS: {sz/1024**3:.2f} GB")
            sys.exit(0)
        else:
            print(f"Removing incomplete file ({sz/1024/1024:.1f} MB)")
            OUTPUT.unlink()

    print(f"\n{'='*60}")
    print(f"Attempt {attempt}/{MAX_RETRIES}")
    print(f"{'='*60}")

    try:
        gdown.download(URL, str(OUTPUT), quiet=False)
    except Exception as e:
        print(f"Error: {e}")

    if OUTPUT.exists():
        sz = OUTPUT.stat().st_size
        gb = sz / 1024**3
        if gb > 3.5:
            print(f"\nSUCCESS! Downloaded {gb:.2f} GB")
            sys.exit(0)
        else:
            print(f"Incomplete: {gb:.2f} GB")

    wait = min(30 + attempt * 2, 120)
    print(f"Waiting {wait}s...")
    time.sleep(wait)

print("FAILED after max retries")
sys.exit(1)
