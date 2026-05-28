"""Model checkpoint save/load utilities."""

from __future__ import annotations

import json
from pathlib import Path

import torch


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    best_metric: float,
    config,
    output_dir: Path,
    filename: str = "checkpoint.pth",
    is_best: bool = False,
):
    """Save model and training state to a checkpoint.

    Args:
        model: Model instance.
        optimizer: Optimizer instance.
        epoch: Current epoch number.
        best_metric: Best validation metric value so far.
        config: Full config namespace.
        output_dir: Directory to save checkpoints.
        filename: Checkpoint filename.
        is_best: Whether this is the best model so far.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
    }

    path = output_dir / filename
    torch.save(checkpoint, path)

    if is_best:
        best_path = output_dir / "best_model.pth"
        torch.save(checkpoint, best_path)

    # Save training state separately for resumption
    state_path = output_dir / "training_state.json"
    state_path.write_text(json.dumps({
        "epoch": epoch,
        "best_metric": best_metric,
    }, indent=2))


def load_checkpoint(
    model,
    optimizer,
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> dict:
    """Load a checkpoint.

    Args:
        model: Model instance.
        optimizer: Optimizer instance (state will be loaded into it).
        checkpoint_path: Path to the checkpoint file.
        device: Device to map tensors to.

    Returns:
        Dict with 'epoch', 'best_metric' from the checkpoint.
    """
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "best_metric": checkpoint.get("best_metric", 0.0),
    }
