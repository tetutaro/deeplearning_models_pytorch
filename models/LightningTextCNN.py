#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from .TextCNN import TextCNN
from .Lightning import ConfigLightning, Lightning

DEFAULT_LOG_CLASS_NAME = 'TextCNN'


class ConfigLightningTextCNN(ConfigLightning):
    lightning_textcnn_params = [
        # name, vtype, is_require, default
        ('log_class_name', str, False, DEFAULT_LOG_CLASS_NAME),
    ]

    def __init__(
        self: ConfigLightningTextCNN,
        config_lightning_json: str
    ) -> Dict:
        # load json
        config = dict()
        self._load_one(config, config_lightning_json)
        # set parameters
        for param in self.lightning_params:
            self._init_param(config, *param)
        for param in self.lightning_textcnn_params:
            self._init_param(config, *param)
        return


class LightningTextCNN(Lightning):
    def __init__(
        self: LightningTextCNN,
        config_lightning_json: str,
        model: TextCNN,
        dataset: TensorDataset
    ) -> None:
        # initialize LightningModule
        super().__init__()
        self.config = ConfigLightningTextCNN(config_lightning_json)
        self.model = model
        self._init_lightinig(dataset)
        return None

    def forward(self: LightningTextCNN, x: torch.Tensor):
        return self.model.forward(x)

    def training_step(
        self: LightningTextCNN,
        batch: Tuple[torch.Tensor],
        batch_index: int
    ) -> Dict:
        x, y = batch
        # optimizer の初期化は Lightning 内で自動的に行われる
        # forward propagation
        output = self.forward(x)
        # 損失の計算
        loss = F.cross_entropy(output, y)
        # 確率が最大になるクラスを取得
        pred = torch.argmax(output, dim=1)
        # 正解数とデータ数
        ndata = torch.tensor(len(y) * 1.0)
        correct = torch.sum(y == pred)
        # Accuracyを計算
        acc = torch.tensor(correct.item() / ndata.item())
        # backward propagation と 勾配の更新は
        # Lightning 内で自動的に行われる
        return {
            'loss': loss,
            'acc': acc,
            'correct': correct,
            'ndata': ndata,
        }

    def training_epoch_end(
        self: LightningTextCNN,
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
        self: LightningTextCNN,
        batch: Tuple[torch.Tensor],
        batch_index: int
    ) -> Dict:
        x, y = batch
        # forward propagation
        output = self.model(x)
        # 損失の計算
        loss = F.cross_entropy(output, y)
        # 確率が最大になるクラスを取得
        pred = torch.argmax(output, dim=1)
        # 正解数とデータ数
        ndata = torch.tensor(len(y) * 1.0)
        correct = torch.sum(y == pred)
        # Accuracyを計算
        acc = torch.tensor(correct.item() / ndata.item())
        return {
            'val_loss': loss,
            'val_acc': acc,
            'val_correct': correct,
            'val_ndata': ndata,
        }

    def validation_epoch_end(
        self: LightningTextCNN,
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
