#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict
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


class ConfigLightning(Config):
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

    def __init__(
        self: ConfigLightning,
        config_lightning_json: str
    ) -> Dict:
        # call parent function
        super().__init__()
        config = dict()
        self.load_one(config, config_lightning_json)
        # set parameters
        for param in self.lightning_params:
            self.init_param(config, *param)
        return config


class Lightning(LightningModule):
    def __init__(
        self: Lightning,
        config: ConfigLightning,
        dataset: TensorDataset
    ) -> None:
        # initialize parent class
        super().__init__()
        self.config = config
        self.dataset = dataset
        # create early stopping instance
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=config.min_delta,
            patience=config.patience
        )
        # create logger instance
        logger = TensorBoardLogger(
            'lightning_logs', name=config.log_class_name
        )
        # detect gpus
        if torch.cuda.is_available():
            # gpus = torch.cuda.device_count()
            # distributed_backend = 'dp'
            gpus = 1
            distributed_backend = None
        else:
            gpus = None
            distributed_backend = None
        # create trainer instance
        self.trainer = Trainer(
            min_epochs=config.min_epochs,
            max_epochs=config.max_epochs,
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
