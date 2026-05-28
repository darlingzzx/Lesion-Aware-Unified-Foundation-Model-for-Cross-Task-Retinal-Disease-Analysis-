"""Learning rate scheduler factory."""

from __future__ import annotations

import torch.optim as optim


def build_scheduler(optimizer, config, t_max: int) -> optim.lr_scheduler.LRScheduler:
    """Create a learning rate scheduler.

    Args:
        optimizer: Optimizer instance.
        config: Training config namespace.
        t_max: Maximum number of iterations/epochs.

    Returns:
        PyTorch LR scheduler.
    """
    scheduler_name = getattr(config, "scheduler", "cosine")

    if scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    elif scheduler_name == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=t_max // 3, gamma=0.1)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
