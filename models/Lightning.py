#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Tuple, Callable, Optional
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
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
DEFAULT_EARLY_STOPPING = True


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
        ('early_stopping', bool, False, DEFAULT_EARLY_STOPPING),
    ]


class Lightning(LightningModule, ABC):
    def _init_lightinig(
        self: Lightning,
        model: nn.Mudule,
        dataset: TensorDataset,
        ckpt_func: Optional[Callable] = None,
        ckpt_dataset: Optional[TensorDataset] = None
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.ckpt_func = ckpt_func
        self.ckpt_dataset = ckpt_dataset
        # create early stopping instance
        if self.config.early_stopping:
            early_stop_callback = EarlyStopping(
                monitor='val_loss',
                min_delta=self.config.min_delta,
                patience=self.config.patience
            )
        else:
            early_stop_callback = None
        # create logger instance
        logger = TensorBoardLogger(
            'lightning_logs', name=self.config.log_class_name
        )
        # detect gpus
        if torch.cuda.is_available():
            gpus = torch.cuda.device_count()
        else:
            gpus = None
        # create trainer instance
        self.trainer = Trainer(
            min_epochs=self.config.min_epochs,
            max_epochs=self.config.max_epochs,
            gpus=gpus,
            distributed_backend=None,
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

    def on_epoch_start(self: Lightning) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

    def on_epoch_end(self: Lightning) -> None:
        if self.ckpt_func is not None:
            self.ckpt_func(
                self.model,
                self.ckpt_dataset,
                self.current_epoch
            )
        return

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
