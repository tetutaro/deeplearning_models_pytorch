#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Tuple
from abc import ABC, abstractmethod
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from .Config import Config

DEFAULT_BATCH_SIZE = 100
DEFAULT_VALIDATION_RATE = 0.25
DEFAULT_NUM_WORKERS = 0
DEFAULT_MIN_EPOCHS = 1
DEFAULT_MAX_EPOCHS = 1000
DEFAULT_MIN_DELTA = 0
DEFAULT_PATIENCE = 0
DEFAULT_LEARNING_RATE = 0.001


class ConfigLightning(Config, ABC):
    lightning_params = [
        # name, vtype, is_require, default
        ('batch_size', int, False, DEFAULT_BATCH_SIZE),
        ('validation_rate', float, False, DEFAULT_VALIDATION_RATE),
        ('num_workers', int, False, DEFAULT_NUM_WORKERS),
        ('min_epochs', int, False, DEFAULT_MIN_EPOCHS),
        ('max_epochs', int, False, DEFAULT_MAX_EPOCHS),
        ('min_delta', float, False, DEFAULT_MIN_DELTA),
        ('patience', int, False, DEFAULT_PATIENCE),
        ('learning_rate', float, False, DEFAULT_LEARNING_RATE),
    ]


class Lightning(LightningModule, ABC):
    def _init_lightinig(
        self: Lightning,
        dataset: TensorDataset
    ) -> None:
        self.dataset = dataset
        # create early stopping instance
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=self.config.min_delta,
            patience=self.config.patience
        )
        # create logger instance
        logger = TensorBoardLogger(
            'lightning_logs', name=self.config.log_class_name
        )
        # detect gpus
        if torch.cuda.is_available():
            gpus = torch.cuda.device_count()
            if gpus == 1:
                distributed_backend = None
            else:
                distributed_backend = 'ddp'
        else:
            gpus = None
            distributed_backend = None
        # create trainer instance
        self.trainer = Trainer(
            min_epochs=self.config.min_epochs,
            max_epochs=self.config.max_epochs,
            gpus=gpus,
            distributed_backend=distributed_backend,
            early_stop_callback=early_stop_callback,
            logger=logger,
            num_sanity_val_steps=1
        )
        return

    def prepare_data(self: Lightning) -> None:
        num_data = len(self.dataset)
        if self.config.validation_rate > 0:
            num_train = int(num_data * (1 - self.config.validation_rate))
            num_val = num_data - num_train
            self.train_dataset, self.val_dataset = random_split(
                self.dataset, [num_train, num_val]
            )
        else:
            self.train_dataset = self.dataset
            self.val_dataset = self.dataset
        return

    def train_dataloader(self: Lightning) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers
        )

    def val_dataloader(self: Lightning) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )

    def configure_optimizers(self: Lightning) -> optim.Optimizer:
        return optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )

    def fit(self: Lightning) -> None:
        return self.trainer.fit(self)

    @abstractmethod
    def forward(self: Lightning, x: torch.Tensor):
        return self.model.forward(x)

    @abstractmethod
    def training_step(
        self: Lightning,
        batch: Tuple[torch.Tensor],
        batch_index: int
    ) -> Dict:
        return {
            'loss': None,
        }

    @abstractmethod
    def training_epoch_end(self: Lightning, outputs: List[Dict]) -> Dict:
        return {
            'loss': None,
        }

    @abstractmethod
    def validation_step(
        self: Lightning,
        batch: Tuple[torch.Tensor],
        batch_index: int
    ) -> Dict:
        return {
            'val_loss': None,
        }

    @abstractmethod
    def validation_epoch_end(self: Lightning, outputs: List[Dict]) -> Dict:
        return {
            'val_loss': None,
        }
