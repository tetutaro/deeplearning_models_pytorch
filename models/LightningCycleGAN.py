#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple, Callable
from itertools import chain
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from torch.utils.data import TensorDataset
from .CycleGAN import CycleGAN
from .Lightning import ConfigLightning, Lightning

DEFAULT_LOG_CLASS_NAME = 'CycleGAN'


class ConfigLightningCycleGAN(ConfigLightning):
    lightning_cyclegan_params = [
        # name, vtype, is_require, default
        ('log_class_name', str, False, DEFAULT_LOG_CLASS_NAME),
    ]

    def __init__(
        self: ConfigLightningCycleGAN,
        config_lightning_json: str
    ) -> Dict:
        # load json
        config = dict()
        self._load_one(config, config_lightning_json)
        # set parameters
        for param in self.lightning_params:
            self._init_param(config, *param)
        for param in self.lightning_cyclegan_params:
            self._init_param(config, *param)
        return


class LightningCycleGAN(Lightning):
    def __init__(
        self: LightningCycleGAN,
        config_lightning_json: str,
        model: CycleGAN,
        dataset: TensorDataset,
        ckpt_func: Callable,
        ckpt_dataset: TensorDataset
    ) -> None:
        # initialize LightningModule
        super().__init__()
        self.config = ConfigLightningCycleGAN(config_lightning_json)
        self._init_lightinig(
            model=model,
            dataset=dataset,
            ckpt_func=ckpt_func,
            ckpt_dataset=ckpt_dataset
        )
        return None

    def configure_optimizers(self: LightningCycleGAN) -> List:
        optim_gen = optim.Adam(chain(
            self.model.genAB.parameters(),
            self.model.genBA.parameters()
        ), lr=self.config.learning_rate, betas=(0.5, 0.999))
        optim_dis = optim.Adam(chain(
            self.model.disA.parameters(),
            self.model.disB.parameters(),
        ), lr=self.config.learning_rate, betas=(0.5, 0.999))
        epochs_decay = self.config.max_epochs - self.config.min_epochs

        def lambda_rule(epoch: int) -> float:
            return 1.0 - (
                max(
                    0, epoch - self.config.min_epochs
                ) / float(epochs_decay + 1)
            )

        sched_gen = sched.LambdaLR(optim_gen, lr_lambda=lambda_rule)
        sched_dis = sched.LambdaLR(optim_dis, lr_lambda=lambda_rule)
        return [
            [optim_gen, optim_dis],
            [sched_gen, sched_dis],
        ]

    def forward(
        self: LightningCycleGAN,
        module: str,
        realA: torch.Tensor,
        realB: torch.Tensor
    ):
        return self.model.forward(
            module=module, realA=realA, realB=realB
        )

    def training_step(
        self: LightningCycleGAN,
        batch: Tuple[torch.Tensor],
        batch_idx: int,
        optimizer_idx: int
    ) -> Dict:
        realA = batch[0]
        realB = batch[1]
        if optimizer_idx == 0:
            loss = self.forward(
                module='gen', realA=realA, realB=realB
            )
        else:
            loss = self.forward(
                module='dis', realA=realA, realB=realB
            )
        return {
            'loss': loss,
        }

    def training_epoch_end(
        self: LightningCycleGAN,
        outputs: List[Dict]
    ) -> Dict:
        avg_loss = torch.stack([
            x['loss'] for x in outputs
        ]).mean()
        tensorboard_logs = {
            'loss': avg_loss,
            'step': self.current_epoch,
        }
        return {
            'loss': avg_loss,
            'log': tensorboard_logs,
        }

    def validation_step(
        self: LightningCycleGAN,
        batch: Tuple[torch.Tensor],
        batch_index: int
    ) -> Dict:
        realA = batch[0]
        realB = batch[1]
        loss_gen = self.forward(
            module='gen', realA=realA, realB=realB
        )
        loss_dis = self.forward(
            module='dis', realA=realA, realB=realB
        )
        loss = loss_gen + loss_dis
        return {
            'val_loss': loss,
        }

    def validation_epoch_end(
        self: LightningCycleGAN,
        outputs: List[Dict]
    ) -> Dict:
        avg_loss = torch.stack([
            x['val_loss'] for x in outputs
        ]).mean()
        tensorboard_logs = {
            'val_loss': avg_loss,
            'step': self.current_epoch,
        }
        return {
            'val_loss': avg_loss,
            'log': tensorboard_logs,
        }
