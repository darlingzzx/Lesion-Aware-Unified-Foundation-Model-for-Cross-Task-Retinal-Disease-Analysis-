"""Model checkpoint save/load utilities."""

from __future__ import annotations

import json
from pathlib import Path

import torch


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_metric: float,
    config,
    output_dir: Path,
    stage_name: str = "stage1",
    filename: str = "checkpoint.pth",
    is_best: bool = False,
):
    """Save model and training state to a checkpoint.

    Args:
        model: Model instance.
        optimizer: Optimizer instance.
        scheduler: LR scheduler instance (can be None).
        epoch: Current epoch number (within the stage).
        best_metric: Best validation metric value so far.
        config: Full config namespace.
        output_dir: Directory to save checkpoints.
        stage_name: Which training stage ("stage1" or "stage2").
        filename: Checkpoint filename.
        is_best: Whether this is the best model so far.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "stage_name": stage_name,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
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
        "stage_name": stage_name,
        "best_metric": best_metric,
    }, indent=2))


def load_checkpoint(
    model,
    optimizer,
    checkpoint_path: str | Path,
    device: torch.device | None = None,
    scheduler=None,
) -> dict:
    """Load a checkpoint.

    Args:
        model: Model instance.
        optimizer: Optimizer instance (state will be loaded into it).
        checkpoint_path: Path to the checkpoint file.
        device: Device to map tensors to.
        scheduler: Optional LR scheduler instance (state will be restored).

    Returns:
        Dict with 'epoch', 'stage_name', 'best_metric' from the checkpoint.
    """
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "stage_name": checkpoint.get("stage_name", "stage1"),
        "best_metric": checkpoint.get("best_metric", 0.0),
    }
