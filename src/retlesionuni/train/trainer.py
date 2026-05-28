"""Two-stage training loop for RetLesionUni."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from ..data.joint_loader import DataLoaderFactory
from ..data.transforms import (
    get_ddr_transform,
    get_odir_eval_transform,
    get_odir_transforms,
)
from ..eval.metrics import compute_ddr_metrics, compute_odir_metrics
from ..losses.asl_loss import AsymmetricLoss
from ..losses.dice_loss import DiceCELoss
from ..models.retlesionuni import RetLesionUni
from ..utils.logger import Logger
from .checkpoint import save_checkpoint
from .scheduler import build_scheduler


class Trainer:
    """Two-stage trainer for RetLesionUni.

    Stage 1: Freeze 70% backbone, train heads only (L_O + L_D).
    Stage 2: Unfreeze all, full loss (L_O + L_D + L_lesion + L_align).
    """

    def __init__(self, config):
        self.config = config
        self.cfg_train = config.training
        self.cfg_model = config.model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        # Create data loaders
        self._setup_data()

        # Create model
        self.model = RetLesionUni(config).to(self.device)
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")

        # Create losses
        self.loss_odir = AsymmetricLoss(
            gamma_pos=config.loss.odir.gamma_pos,
            gamma_neg=config.loss.odir.gamma_neg,
        )
        self.loss_ddr = DiceCELoss(
            dice_weight=config.loss.ddr.dice_weight,
            ce_weight=config.loss.ddr.ce_weight,
        )

        # Logger
        self.logger = Logger(
            log_dir=config.logging.log_dir,
            enabled=config.logging.tensorboard,
        )

        # Checkpoint dir
        self.ckpt_dir = Path(config.output_dir) / "checkpoints" / config.exp_name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _setup_data(self):
        factory = DataLoaderFactory(self.config)

        # ODIR transforms
        odir_tr_v1, odir_tr_v2 = get_odir_transforms(self.cfg_model.img_size)
        odir_eval_tr = get_odir_eval_transform(self.cfg_model.img_size)

        # DDR transforms
        ddr_tr = get_ddr_transform(self.cfg_model.img_size, is_train=True)
        ddr_eval_tr = get_ddr_transform(self.cfg_model.img_size, is_train=False)

        self.odir_train_ds, self.odir_valid_loader, self.odir_test_loader = factory.create_odir_loaders(
            odir_tr_v1, odir_tr_v2, odir_eval_tr
        )
        self.ddr_train_ds, self.ddr_valid_loader, self.ddr_test_loader = factory.create_ddr_loaders(
            ddr_tr, ddr_eval_tr
        )

        self.joint_train_loader = factory.create_joint_train_loader(
            self.odir_train_ds, self.ddr_train_ds
        )

    def train(self):
        """Run full two-stage training."""
        total_epochs = self.cfg_train.stage1.epochs + self.cfg_train.stage2.epochs
        print(f"\n{'='*60}")
        print(f"Training: {self.cfg_train.stage1.epochs} (stage1) + {self.cfg_train.stage2.epochs} (stage2) = {total_epochs} epochs")
        print(f"{'='*60}")

        # --- Stage 1 ---
        print("\n>>> Stage 1: Freeze backbone, train heads only")
        self._run_stage("stage1")

        # --- Stage 2 ---
        print("\n>>> Stage 2: Unfreeze all, full loss with LPM + CTAM")
        self._run_stage("stage2")

        # Save final model
        save_checkpoint(
            self.model, self.optimizer,
            epoch=total_epochs,
            best_metric=self.best_metric,
            config=self.config,
            output_dir=self.ckpt_dir,
            filename="final_model.pth",
            is_best=False,
        )
        self.logger.close()
        print("\nTraining complete!")

    def _run_stage(self, stage_name: str):
        stage_cfg = getattr(self.cfg_train, stage_name)
        epochs = stage_cfg.epochs
        self.best_metric = 0.0 if self.cfg_train.checkpoint.monitor_mode == "max" else float("inf")

        # Configure model for stage
        self.model.set_stage_config(
            use_lpm=stage_cfg.use_lpm,
            use_ctam=stage_cfg.use_ctam,
            warmup_factor=1.0,
        )

        # Freeze/unfreeze encoder
        if stage_cfg.freeze_ratio > 0:
            self.model.encoder.freeze_layers(stage_cfg.freeze_ratio)
        else:
            self.model.encoder.unfreeze_all()

        # Optimizer: only trainable params
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=stage_cfg.lr,
            betas=tuple(self.cfg_train.betas),
            weight_decay=self.cfg_train.weight_decay,
        )

        # Scheduler
        self.scheduler = build_scheduler(self.optimizer, self.cfg_train, t_max=epochs)

        for epoch in range(1, epochs + 1):
            # Update warmup factor for stage 2
            if stage_name == "stage2" and stage_cfg.use_ctam:
                warmup = min(1.0, epoch / max(1, stage_cfg.warmup_epochs))
                self.model.set_stage_config(
                    use_lpm=stage_cfg.use_lpm,
                    use_ctam=stage_cfg.use_ctam,
                    warmup_factor=warmup,
                )

            train_loss = self._train_epoch(epoch, epochs, stage_name)
            self.scheduler.step()

            # Validation
            odir_metrics = self._validate_odir()
            ddr_metrics = self._validate_ddr()

            # Logging
            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch}/{epochs} | "
                f"LR={lr:.2e} | "
                f"Train Loss={train_loss:.4f} | "
                f"ODIR F1={odir_metrics.get('f1', 0):.4f} | "
                f"DDR mDice={ddr_metrics.get('mDice', 0):.4f}"
            )

            if self.logger.enabled:
                self.logger.log_scalar("train/loss", train_loss, epoch)
                self.logger.log_scalar("train/lr", lr, epoch)
                for k, v in {**odir_metrics, **ddr_metrics}.items():
                    self.logger.log_scalar(f"val/{k}", v, epoch)

            # Checkpointing
            monitor_val = ddr_metrics.get(
                self.cfg_train.checkpoint.monitor_metric.replace("ddr_", ""), 0
            )
            is_best = (
                (self.cfg_train.checkpoint.monitor_mode == "max" and monitor_val > self.best_metric)
                or (self.cfg_train.checkpoint.monitor_mode == "min" and monitor_val < self.best_metric)
            )
            if is_best:
                self.best_metric = monitor_val

            if (self.cfg_train.checkpoint.save_every_n_epochs > 0
                    and epoch % self.cfg_train.checkpoint.save_every_n_epochs == 0):
                save_checkpoint(
                    self.model, self.optimizer,
                    epoch=epoch, best_metric=self.best_metric,
                    config=self.config, output_dir=self.ckpt_dir,
                    filename=f"{stage_name}_epoch_{epoch}.pth",
                    is_best=is_best,
                )

    def _train_epoch(self, epoch: int, total_epochs: int, stage_name: str) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.joint_train_loader)

        for batch_idx, batch in enumerate(self.joint_train_loader):
            batch = self._to_device(batch)

            self.optimizer.zero_grad()

            outputs = self.model(batch)

            loss_odir = self.loss_odir(
                outputs["odir_logits"],
                batch["odir"]["labels"],
            )
            loss_ddr = self.loss_ddr(
                outputs["ddr_seg"],
                batch["ddr"]["mask"],
            )

            loss = loss_odir + loss_ddr + outputs["loss_lesion"] + outputs["loss_align"]
            loss.backward()

            if self.cfg_train.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg_train.gradient_clip
                )

            self.optimizer.step()
            total_loss += loss.item()

            if batch_idx % self.config.logging.print_interval == 0:
                print(
                    f"  Batch {batch_idx}/{num_batches} | "
                    f"L_O={loss_odir.item():.4f} L_D={loss_ddr.item():.4f} "
                    f"L_les={outputs['loss_lesion'].item():.4f} "
                    f"L_align={outputs['loss_align'].item():.4f}"
                )

        return total_loss / max(num_batches, 1)

    def _validate_odir(self) -> dict:
        self.model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in self.odir_valid_loader:
                batch = self._to_device(batch)
                # For validation, use single forward pass through encoder
                enc = self.model.encoder(batch["image_v1"])
                preds = self.model.odir_head(enc["cls_token"])
                all_preds.append(preds.cpu())
                all_targets.append(batch["labels"].cpu())
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()
        return compute_odir_metrics(all_preds, all_targets)

    def _validate_ddr(self) -> dict:
        self.model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in self.ddr_valid_loader:
                batch = self._to_device(batch)
                enc = self.model.encoder(batch["image"], return_multilevel=True)
                preds = self.model.ddr_head(
                    enc["multilevel_features"], self.cfg_model.img_size
                )
                all_preds.append(preds.cpu())
                all_targets.append(batch["mask"].cpu())
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()
        return compute_ddr_metrics(all_preds, all_targets)

    def _to_device(self, batch: dict) -> dict:
        """Recursively move tensors in a batch dict to the device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        if isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        if isinstance(batch, list):
            return [self._to_device(v) for v in batch]
        return batch
