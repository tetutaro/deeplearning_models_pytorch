#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from .BertSum import BertSum
from .Lightning import ConfigLightning, Lightning

DEFAULT_LOG_CLASS_NAME = 'BertSum'


class ConfigLightningBertSum(ConfigLightning):
    lightning_bertsum_params = [
        # name, vtype, is_require, default
        ('log_class_name', str, False, DEFAULT_LOG_CLASS_NAME),
    ]

    def __init__(
        self: ConfigLightningBertSum,
        config_lightning_json: str
    ) -> Dict:
        # load json
        config = dict()
        self._load_one(config, config_lightning_json)
        # set parameters
        for param in self.lightning_params:
            self._init_param(config, *param)
        for param in self.lightning_bertsum_params:
            self._init_param(config, *param)
        return


class LightningBertSum(Lightning):
    def __init__(
        self: LightningBertSum,
        config_lightning_json: str,
        model: BertSum,
        dataset: TensorDataset
    ) -> None:
        # initialize LightningModule
        super().__init__()
        self.config = ConfigLightningBertSum(config_lightning_json)
        self._init_lightinig(model=model, dataset=dataset)
        return None

    def forward(
        self: LightningBertSum,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        segment_ids: torch.Tensor,
        cls_ids: torch.Tensor,
        cls_ids_mask: torch.Tensor,
        cls_ids_mask_bool: torch.Tensor
    ):
        return self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            cls_ids=cls_ids,
            cls_ids_mask=cls_ids_mask,
            cls_ids_mask_bool=cls_ids_mask_bool
        )

    def training_step(
        self: LightningBertSum,
        batch: Tuple[torch.Tensor],
        batch_index: int
    ) -> Dict:
        scores = self.forward(
            input_ids=batch[0],
            attention_mask=batch[1],
            segment_ids=batch[2],
            cls_ids=batch[3],
            cls_ids_mask=batch[4],
            cls_ids_mask_bool=batch[5]
        )
        cls_ids_mask = batch[4]
        labels = batch[6]
        loss = F.binary_cross_entropy(scores, labels)
        loss = (loss * cls_ids_mask).sum()
        return {
            'loss': loss,
        }

    def training_epoch_end(
        self: LightningBertSum,
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
        self: LightningBertSum,
        batch: Tuple[torch.Tensor],
        batch_index: int
    ) -> Dict:
        scores = self.forward(
            input_ids=batch[0],
            attention_mask=batch[1],
            segment_ids=batch[2],
            cls_ids=batch[3],
            cls_ids_mask=batch[4],
            cls_ids_mask_bool=batch[5]
        )
        cls_ids_mask = batch[4]
        labels = batch[6]
        loss = F.binary_cross_entropy(scores, labels)
        loss = (loss * cls_ids_mask).sum()
        return {
            'val_loss': loss,
        }

    def validation_epoch_end(
        self: LightningBertSum,
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
