"""Joint batch sampler for ODIR + DDR training."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


class JointDataLoader:
    """Iterates over ODIR and DDR datasets jointly, yielding combined batches.

    ODIR uses weighted random sampling to ensure DR-positive samples per batch.
    """

    def __init__(
        self,
        odir_dataset,
        ddr_dataset,
        batch_size_odir: int = 8,
        batch_size_ddr: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        dr_positive_weight: float = 3.0,
        drop_last: bool = True,
    ):
        self.odir_dataset = odir_dataset
        self.ddr_dataset = ddr_dataset

        # ODIR: weighted sampling for DR-positive
        odir_weights = odir_dataset.get_sample_weights(dr_positive_weight)
        odir_sampler = WeightedRandomSampler(
            weights=odir_weights.tolist(),
            num_samples=len(odir_dataset),
            replacement=True,
        )

        self.odir_loader = DataLoader(
            odir_dataset,
            batch_size=batch_size_odir,
            sampler=odir_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        self.ddr_loader = DataLoader(
            ddr_dataset,
            batch_size=batch_size_ddr,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

        self._odir_iter = iter(self.odir_loader)
        self._ddr_iter = iter(self.ddr_loader)

    def __iter__(self):
        self._odir_iter = iter(self.odir_loader)
        self._ddr_iter = iter(self.ddr_loader)
        return self

    def __next__(self) -> dict:
        odir_batch = next(self._odir_iter)
        ddr_batch = next(self._ddr_iter)
        return {"odir": odir_batch, "ddr": ddr_batch}

    def __len__(self) -> int:
        return min(len(self.odir_loader), len(self.ddr_loader))


class DataLoaderFactory:
    """Creates DataLoaders and JointDataLoader from config and preprocessed cache."""

    def __init__(self, config):
        self.config = config
        self.cfg_data = config.data
        self.cfg_train = config.training
        self.cfg_model = config.model

    def create_odir_loaders(self, transform_v1, transform_v2, eval_transform):
        """Create train and validation ODIR DataLoaders."""
        from .odir_dataset import ODIRDataset

        odir_cache = str(Path(self.cfg_data.preprocessed_cache) / "odir")
        train_ds = ODIRDataset(odir_cache, "train", transform_v1, transform_v2)
        valid_ds = ODIRDataset(odir_cache, "valid", eval_transform, None)
        test_ds = ODIRDataset(odir_cache, "test", eval_transform, None)

        valid_loader = DataLoader(
            valid_ds,
            batch_size=self.cfg_train.batch_size_odir,
            shuffle=False,
            num_workers=self.cfg_train.num_workers,
            pin_memory=self.cfg_train.pin_memory,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.cfg_train.batch_size_odir,
            shuffle=False,
            num_workers=self.cfg_train.num_workers,
            pin_memory=self.cfg_train.pin_memory,
        )

        return train_ds, valid_loader, test_loader

    def create_ddr_loaders(self, train_transform, eval_transform):
        """Create train and validation DDR DataLoaders."""
        from .ddr_dataset import DDRDataset

        ddr_cache = str(Path(self.cfg_data.preprocessed_cache) / "ddr")
        train_ds = DDRDataset(ddr_cache, "train", train_transform)
        valid_ds = DDRDataset(ddr_cache, "valid", eval_transform)
        test_ds = DDRDataset(ddr_cache, "test", eval_transform)

        valid_loader = DataLoader(
            valid_ds,
            batch_size=self.cfg_train.batch_size_ddr,
            shuffle=False,
            num_workers=self.cfg_train.num_workers,
            pin_memory=self.cfg_train.pin_memory,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.cfg_train.batch_size_ddr,
            shuffle=False,
            num_workers=self.cfg_train.num_workers,
            pin_memory=self.cfg_train.pin_memory,
        )

        return train_ds, valid_loader, test_loader

    def create_joint_train_loader(self, odir_train_ds, ddr_train_ds) -> JointDataLoader:
        """Create the joint training loader with weighted ODIR sampling."""
        return JointDataLoader(
            odir_dataset=odir_train_ds,
            ddr_dataset=ddr_train_ds,
            batch_size_odir=self.cfg_train.batch_size_odir,
            batch_size_ddr=self.cfg_train.batch_size_ddr,
            num_workers=self.cfg_train.num_workers,
            pin_memory=self.cfg_train.pin_memory,
            dr_positive_weight=self.cfg_train.dr_positive_weight,
            drop_last=True,
        )
