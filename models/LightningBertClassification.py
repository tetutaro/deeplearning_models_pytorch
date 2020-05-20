#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple
import torch
from torch.utils.data import TensorDataset
from .BertClassification import BertClassification
from .Lightning import ConfigLightning, Lightning

DEFAULT_LOG_CLASS_NAME = 'BertClassification'


class ConfigLightningBertClassification(ConfigLightning):
    lightning_bert_classification_params = [
        # name, vtype, is_require, default
        ('log_class_name', str, False, DEFAULT_LOG_CLASS_NAME),
    ]

    def __init__(
        self: ConfigLightningBertClassification,
        config_lightning_json: str
    ) -> Dict:
        # load json
        config = dict()
        self._load_one(config, config_lightning_json)
        # set parameters
        for param in self.lightning_params:
            self._init_param(config, *param)
        for param in self.lightning_bert_classification_params:
            self._init_param(config, *param)
        return


class LightningBertClassification(Lightning):
    def __init__(
        self: LightningBertClassification,
        config_lightning_json: str,
        model: BertClassification,
        dataset: TensorDataset
    ) -> None:
        # initialize LightningModule
        super().__init__()
        self.config = ConfigLightningBertClassification(config_lightning_json)
        self.model = model
        self._init_lightinig(dataset)
        return None

    def forward(self: LightningBertClassification, x: Tuple[torch.Tensor]):
        return self.model.forward(x)

    def training_step(
        self: LightningBertClassification,
        batch: Tuple[torch.Tensor],
        batch_index: int
    ) -> Dict:
        output = self.forward(batch)
        loss = output[0]
        logit = output[1]
        labels = batch[3]
        pred = torch.argmax(logit, axis=1)
        ndata = torch.tensor(len(labels) * 1.0)
        correct = torch.sum(labels == pred)
        acc = torch.tensor(correct.item() / ndata.item())
        return {
            'loss': loss,
            'acc': acc,
            'correct': correct,
            'ndata': ndata,
        }

    def training_epoch_end(
        self: LightningBertClassification,
        outputs: List[Dict]
    ) -> Dict:
        avg_loss = torch.stack([
            x['loss'] for x in outputs
        ]).mean()
        sum_correct = torch.stack([
            x['correct'] for x in outputs
        ]).sum()
        sum_ndata = torch.stack([
            x['ndata'] for x in outputs
        ]).sum()
        avg_acc = torch.tensor(sum_correct.item() / sum_ndata.item())
        tensorboard_logs = {
            'loss': avg_loss,
            'acc': avg_acc,
            'step': self.current_epoch,
        }
        return {
            'loss': avg_loss,
            'log': tensorboard_logs,
        }

    def validation_step(
        self: LightningBertClassification,
        batch: Tuple[torch.Tensor],
        batch_index: int
    ) -> Dict:
        output = self.forward(batch)
        loss = output[0]
        logit = output[1]
        labels = batch[3]
        pred = torch.argmax(logit, axis=1)
        ndata = torch.tensor(len(labels) * 1.0)
        correct = torch.sum(labels == pred)
        acc = torch.tensor(correct.item() / ndata.item())
        return {
            'val_loss': loss,
            'val_acc': acc,
            'val_correct': correct,
            'val_ndata': ndata,
        }

    def validation_epoch_end(
        self: LightningBertClassification,
        outputs: List[Dict]
    ) -> Dict:
        avg_loss = torch.stack([
            x['val_loss'] for x in outputs
        ]).mean()
        sum_correct = torch.stack([
            x['val_correct'] for x in outputs
        ]).sum()
        sum_ndata = torch.stack([
            x['val_ndata'] for x in outputs
        ]).sum()
        avg_acc = torch.tensor(sum_correct.item() / sum_ndata.item())
        tensorboard_logs = {
            'val_loss': avg_loss,
            'val_acc': avg_acc,
            'step': self.current_epoch,
        }
        return {
            'val_loss': avg_loss,
            'log': tensorboard_logs,
        }
