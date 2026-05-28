"""TensorBoard logger for RetLesionUni."""

from __future__ import annotations

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Wrapper around TensorBoard SummaryWriter for metric logging."""

    def __init__(self, log_dir: str | Path, enabled: bool = True):
        self.enabled = enabled
        self.writer = SummaryWriter(log_dir=str(log_dir)) if enabled else None
        self.global_step = 0

    def log_scalar(self, tag: str, value: float, step: int | None = None):
        if self.writer is not None:
            s = step if step is not None else self.global_step
            self.writer.add_scalar(tag, value, s)

    def log_scalars(self, tag: str, value_dict: dict, step: int | None = None):
        if self.writer is not None:
            s = step if step is not None else self.global_step
            self.writer.add_scalars(tag, value_dict, s)

    def increment_step(self):
        self.global_step += 1

    def close(self):
        if self.writer is not None:
            self.writer.close()
