#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict
import os
import numpy as np
import torch
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification
from .DeepLearning import ConfigDeepLearning

DEFAULT_RANDOM_STATE = None


class ConfigBertClassification(ConfigDeepLearning):
    bert_classification_params = [
        # name, vtype, is_require, default
        ('nict_bert_dir', str, True, None),
        ('num_class', int, True, None),
        ('unique_categories', [list, str], True, None),
        ('random_state', int, False, DEFAULT_RANDOM_STATE),
    ]

    def __init__(
        self: ConfigBertClassification,
        config: Dict,
        config_model_json: str
    ) -> None:
        # model_name
        config['model_name'] = 'BertClassification'
        # load json
        self._load_one(config, config_model_json)
        # init parent class
        self._init_deeplearning(config)
        # set parameters
        for param in self.deeplearning_params:
            self._init_param(config, *param)
        for param in self.bert_classification_params:
            self._init_param(config, *param)
        # value_assertion
        assert(os.path.exists(self.nict_bert_dir))
        return

    def load(self: ConfigBertClassification) -> None:
        return

    def save(self: ConfigBertClassification) -> None:
        return


class BertClassification(nn.Module):
    def __init__(
        self: BertClassification,
        config: Dict,
        config_model_json: str
    ) -> None:
        super().__init__()
        self.config = ConfigBertClassification(config, config_model_json)
        # set random seed
        if self.config.random_state is not None:
            np.random.seed(self.config.random_state)
            torch.manual_seed(self.config.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.config.random_state)
        # load BertForSequenceClassification of NICT BERT
        bert_config = BertConfig.from_pretrained(
            os.path.join(self.config.nict_bert_dir, 'config.json')
        )
        id2label = {i: x for i, x in enumerate(self.config.unique_categories)}
        label2id = {x: i for i, x in enumerate(self.config.unique_categories)}
        bert_config.update({'num_labels': self.config.num_class})
        bert_config.update({'id2label': id2label, 'label2id': label2id})
        if torch.cuda.is_available():
            map_location = torch.device('cuda')
        else:
            map_location = torch.device('cpu')
        state_dict = torch.load(
            os.path.join(self.config.nict_bert_dir, 'pytorch_model.bin'),
            map_location=map_location
        )
        self.model = BertForSequenceClassification.from_pretrained(
            None,
            config=bert_config,
            state_dict=state_dict,
            from_tf=False
        )
        # disable training (changing its weights) of all bert layers
        for name, param in self.model.bert.named_parameters():
            param.requires_grad_(False)
        return

    def forward(
        self: BertClassification,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        return self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )

    def load(self: BertClassification) -> None:
        bert_config = BertConfig.from_pretrained(
            self.config.config_json
        )
        if torch.cuda.is_available():
            map_location = torch.device('cuda')
        else:
            map_location = torch.device('cpu')
        state_dict = torch.load(
            self.config.binary_path, map_location=map_location
        )
        self.model = BertForSequenceClassification.from_pretrained(
            None,
            config=bert_config,
            state_dict=state_dict,
            from_tf=False
        )
        # disable training (changing its weights) of all bert layers
        for name, param in self.model.bert.named_parameters():
            param.requires_grad_(False)
        return

    def save(self: BertClassification) -> None:
        self.model.save_pretrained(self.config.base_dir)
        return

    def train(self: BertClassification, mode: bool) -> None:
        self.model.train(mode)
        return

    def eval(self: BertClassification) -> None:
        self.model.eval()
        return
