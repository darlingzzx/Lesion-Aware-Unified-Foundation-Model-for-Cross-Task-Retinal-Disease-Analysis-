"""Two-stage training loop for RetLesionUni."""

from __future__ import annotations

from pathlib import Path

import torch

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
from .checkpoint import load_checkpoint, save_checkpoint
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
        print(f"Device: {self.device}", flush=True)

        # Create data loaders
        self._setup_data()

        # Create model
        self.model = RetLesionUni(config).to(self.device)
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}", flush=True)

        # Create losses
        self.loss_odir = AsymmetricLoss(
            gamma_pos=config.loss.odir.gamma_pos,
            gamma_neg=config.loss.odir.gamma_neg,
        )
        self.loss_ddr = DiceCELoss(
            dice_weight=config.loss.ddr.dice_weight,
            ce_weight=config.loss.ddr.ce_weight,
            class_weights=getattr(config.loss.ddr, 'class_weights', None),
        )

        # Logger
        self.logger = Logger(
            log_dir=config.logging.log_dir,
            enabled=config.logging.tensorboard,
        )

        # Mixed precision
        self.use_amp = self.cfg_train.mixed_precision in ("fp16", "bf16")
        self.amp_dtype = torch.bfloat16 if self.cfg_train.mixed_precision == "bf16" else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.cfg_train.mixed_precision == "fp16"))
        if self.use_amp:
            print(f"Mixed precision: {self.cfg_train.mixed_precision}")

        # Validation frequency (validate every N epochs to save time)
        self.val_freq = getattr(self.cfg_train, 'val_freq', 5)

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
        """Run full two-stage training with resume support."""
        total_epochs = self.cfg_train.stage1.epochs + self.cfg_train.stage2.epochs
        print(f"\n{'='*60}", flush=True)
        print(f"Training: {self.cfg_train.stage1.epochs} (stage1) + {self.cfg_train.stage2.epochs} (stage2) = {total_epochs} epochs", flush=True)
        print(f"{'='*60}", flush=True)

        # --- Check for resume ---
        resume_path = self.ckpt_dir / "resume_checkpoint.pth"
        enable_resume = getattr(self.cfg_train.checkpoint, 'resume', True)

        if enable_resume and resume_path.exists():
            print(f"\n>>> Resume checkpoint found: {resume_path}", flush=True)
            # Peek at stage/epoch without loading into model yet
            resume_ckpt = torch.load(str(resume_path), map_location=self.device, weights_only=False)
            resumed_stage = resume_ckpt.get("stage_name", "stage1")
            resumed_epoch = resume_ckpt.get("epoch", 0)
            print(f">>> Resuming from: {resumed_stage} epoch {resumed_epoch}", flush=True)

            if resumed_stage == "stage2":
                print(">>> Stage 1 already completed, skipping...", flush=True)
                print("\n>>> Stage 2: Unfreeze all, full loss with LPM + CTAM (resuming)", flush=True)
                self._run_stage("stage2", start_epoch=resumed_epoch + 1, resume_path=resume_path)
            elif resumed_stage == "stage1":
                if resumed_epoch >= self.cfg_train.stage1.epochs:
                    print(">>> Stage 1 completed, moving to Stage 2...", flush=True)
                    print("\n>>> Stage 2: Unfreeze all, full loss with LPM + CTAM", flush=True)
                    self._run_stage("stage2")
                else:
                    print("\n>>> Stage 1: Freeze backbone, train heads only (resuming)", flush=True)
                    self._run_stage("stage1", start_epoch=resumed_epoch + 1, resume_path=resume_path)
                    if resumed_epoch + 1 <= self.cfg_train.stage1.epochs:
                        print("\n>>> Stage 2: Unfreeze all, full loss with LPM + CTAM", flush=True)
                        self._run_stage("stage2")
        else:
            # --- Stage 1 (fresh start) ---
            print("\n>>> Stage 1: Freeze backbone, train heads only", flush=True)
            self._run_stage("stage1")

            # --- Stage 2 ---
            print("\n>>> Stage 2: Unfreeze all, full loss with LPM + CTAM", flush=True)
            self._run_stage("stage2")

        # Save final model
        save_checkpoint(
            self.model, self.optimizer, self.scheduler,
            epoch=total_epochs,
            best_metric=self.best_metric,
            config=self.config,
            output_dir=self.ckpt_dir,
            stage_name="done",
            filename="final_model.pth",
            is_best=False,
        )
        self.logger.close()
        print("\nTraining complete!")

    def _run_stage(self, stage_name: str, start_epoch: int = 1, resume_path: str | None = None):
        stage_cfg = getattr(self.cfg_train, stage_name)
        epochs = stage_cfg.epochs

        if start_epoch > epochs:
            print(f"  Stage {stage_name} already complete (epoch {start_epoch} > {epochs}), skipping.", flush=True)
            return

        resume_ckpt = None
        if resume_path is not None:
            resume_ckpt = torch.load(str(resume_path), map_location=self.device, weights_only=False)
            self.best_metric = resume_ckpt.get("best_metric", 0.0)
        else:
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

        # Resume: restore model/optimizer/scheduler state
        if resume_ckpt is not None:
            self.model.load_state_dict(resume_ckpt["model_state_dict"])
            if resume_ckpt.get("optimizer_state_dict"):
                self.optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
            if resume_ckpt.get("scheduler_state_dict"):
                self.scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
            print(f"  Loaded model, optimizer, and scheduler state from checkpoint.", flush=True)

        for epoch in range(start_epoch, epochs + 1):
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

            # Validation (every N epochs, always on epoch 1 and final epoch)
            do_val = (epoch == 1 or epoch == epochs or epoch % self.val_freq == 0)
            if do_val:
                odir_metrics = self._validate_odir()
                ddr_metrics = self._validate_ddr()
            else:
                odir_metrics = {"f1": 0.0, "accuracy": 0.0, "auc": 0.0}
                ddr_metrics = {"mDice": 0.0, "mIoU": 0.0}

            # Logging
            lr = self.optimizer.param_groups[0]["lr"]
            odir_f1 = odir_metrics.get('f1', 0)
            ddr_dice = ddr_metrics.get('mDice', 0)
            ddr_lesion = ddr_metrics.get('lesion_mDice', 0)
            print(
                f"Epoch {epoch}/{epochs} | "
                f"LR={lr:.2e} | "
                f"Train Loss={train_loss:.4f} | "
                f"ODIR F1={odir_f1:.4f} | "
                f"DDR mDice={ddr_dice:.4f} (lesion={ddr_lesion:.4f})"
                f"{' (skipped val)' if not do_val else ''}",
                flush=True,
            )
            # Per-class ODIR F1 breakdown on validation epochs
            if do_val and 'per_class_f1' in odir_metrics:
                names = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
                pcf1 = odir_metrics['per_class_f1']
                pcf1_str = ' | '.join(f'{n}={v:.3f}' for n, v in zip(names, pcf1))
                print(f"  Per-class F1: {pcf1_str}", flush=True)

            if self.logger.enabled:
                base_epoch = epoch
                if stage_name == "stage2":
                    base_epoch += self.cfg_train.stage1.epochs
                self.logger.log_scalar("train/loss", train_loss, base_epoch)
                self.logger.log_scalar("train/lr", lr, base_epoch)
                for k, v in {**odir_metrics, **ddr_metrics}.items():
                    if isinstance(v, (int, float)):
                        self.logger.log_scalar(f"val/{k}", v, base_epoch)
                    elif isinstance(v, list):
                        for i, cls_v in enumerate(v):
                            self.logger.log_scalar(f"val/{k}_{i}", cls_v, base_epoch)

            # Update best metric and save periodic checkpoint
            if do_val:
                monitor_val = ddr_metrics.get(
                    self.cfg_train.checkpoint.monitor_metric.replace("ddr_", ""), 0
                )
                is_best = (
                    (self.cfg_train.checkpoint.monitor_mode == "max" and monitor_val > self.best_metric)
                    or (self.cfg_train.checkpoint.monitor_mode == "min" and monitor_val < self.best_metric)
                )
                if is_best:
                    self.best_metric = monitor_val
            else:
                is_best = False

            if (self.cfg_train.checkpoint.save_every_n_epochs > 0
                    and epoch % self.cfg_train.checkpoint.save_every_n_epochs == 0):
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    epoch=epoch, best_metric=self.best_metric,
                    config=self.config, output_dir=self.ckpt_dir,
                    stage_name=stage_name,
                    filename=f"{stage_name}_epoch_{epoch}.pth",
                    is_best=is_best,
                )

            # Always save resume checkpoint (for crash recovery)
            save_checkpoint(
                self.model, self.optimizer, self.scheduler,
                epoch=epoch, best_metric=self.best_metric,
                config=self.config, output_dir=self.ckpt_dir,
                stage_name=stage_name,
                filename="resume_checkpoint.pth",
                is_best=False,
            )

    def _train_epoch(self, epoch: int, total_epochs: int, stage_name: str) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.joint_train_loader)
        accum_steps = getattr(self.cfg_train, 'gradient_accumulation_steps', 1)

        self.optimizer.zero_grad()
        for batch_idx, batch in enumerate(self.joint_train_loader):
            batch = self._to_device(batch)

            with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                outputs = self.model(batch)

                loss_odir = self.loss_odir(
                    outputs["odir_logits"],
                    batch["odir"]["labels"],
                )
                loss_ddr = self.loss_ddr(
                    outputs["ddr_seg"],
                    batch["ddr"]["mask"],
                )

                loss = (loss_odir + loss_ddr + outputs["loss_lesion"] + outputs["loss_align"])
                loss = loss / accum_steps  # Normalize for gradient accumulation

            self.scaler.scale(loss).backward()
            total_loss += loss.item() * accum_steps  # Un-normalize for logging

            # Step only after accumulation
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == num_batches:
                self.scaler.unscale_(self.optimizer)

                if self.cfg_train.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg_train.gradient_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            if batch_idx % self.config.logging.print_interval == 0:
                print(
                    f"  Batch {batch_idx}/{num_batches} | "
                    f"L_O={loss_odir.item():.4f} L_D={loss_ddr.item():.4f} "
                    f"L_les={outputs['loss_lesion'].item():.4f} "
                    f"L_align={outputs['loss_align'].item():.4f}",
                    flush=True,
                )

        return total_loss / max(num_batches, 1)

    def _validate_odir(self) -> dict:
        self.model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in self.odir_valid_loader:
                batch = self._to_device(batch)
                # Match training feature path: cls_token + patch_pooled = 2048-dim
                enc = self.model.encoder(batch["image_v1"], return_attention=True)
                patch_pooled = enc["patch_tokens"].mean(dim=1)  # (B, 1024)
                f_odir = torch.cat([enc["cls_token"], patch_pooled], dim=-1)  # (B, 2048)
                # Match training CTAM path: zero input → learned bias for consistent distribution
                if self.model.ctam_fusion is not None:
                    zero_z = torch.zeros(
                        f_odir.shape[0], self.cfg_model.ctam.proj_dim,
                        device=f_odir.device, dtype=f_odir.dtype,
                    )
                    ctam_default = self.model.ctam_fusion_scale * self.model.ctam_fusion(zero_z)
                    f_odir = f_odir + ctam_default
                preds = self.model.odir_head(f_odir)
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
