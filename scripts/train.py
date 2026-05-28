#!/usr/bin/env python
"""Training entry point for RetLesionUni.

Usage:
    python scripts/train.py                          # Use configs/default.yaml
    python scripts/train.py --config configs/test.yaml
    python scripts/train.py --overrides training.stage1.epochs=5 training.batch_size_odir=4
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

from retlesionuni.config import load_config, parse_args
from retlesionuni.train.trainer import Trainer


def main():
    config_path, overrides = parse_args()
    print(f"Loading config: {config_path}")
    config = load_config(config_path, overrides)

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
