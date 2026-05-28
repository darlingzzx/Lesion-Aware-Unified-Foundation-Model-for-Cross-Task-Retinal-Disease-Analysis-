#!/usr/bin/env python
"""Data preprocessing CLI for RetLesionUni.

Usage:
    python scripts/preprocess.py                          # Uses configs/default.yaml
    python scripts/preprocess.py --config configs/test.yaml
    python scripts/preprocess.py --force                  # Regenerates cache
    python scripts/preprocess.py --overrides data.root=data
"""

import sys
from pathlib import Path

# Add src/ to path for direct script invocation
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

from retlesionuni.config import load_config, parse_args
from retlesionuni.data.preprocess_runner import run_preprocessing


def main():
    # Handle --force before arg parsing
    force = "--force" in sys.argv
    if force:
        sys.argv.remove("--force")

    config_path, overrides = parse_args()

    print(f"Loading config: {config_path}")
    config = load_config(config_path, overrides)

    cache_dir = Path(config.data.preprocessed_cache)
    if cache_dir.exists() and not force:
        # Check if metadata files exist
        has_data = False
        for split in config.data.splits:
            ddr_meta = cache_dir / "ddr" / split / "metadata.csv"
            odir_meta = cache_dir / "odir" / split / "metadata.csv"
            if ddr_meta.exists() or odir_meta.exists():
                has_data = True
                break

        if has_data:
            print(f"Cache already exists at: {cache_dir}")
            print("Use --force to regenerate. Skipping preprocessing.")
            return

    print(f"Cache dir: {cache_dir}")
    print(f"Force: {force}")
    print()

    run_preprocessing(config.data, cache_dir)

    print("\nPreprocessing done. Cache is ready for training.")


if __name__ == "__main__":
    main()
