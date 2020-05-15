#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict
import os
import torch.nn as nn
from .Config import Config


class ConfigDeepLearning(Config):
    deeplearning_params = [
        # name, vtype, is_require, default
        ('config_json', str, True, None),
        ('binary_path', str, True, None),
    ]

    def __init__(
        self: ConfigDeepLearning,
        config: Dict,
        config_model_json: str
    ) -> None:
        # call parent function
        super().__init__()
        self.load_one(config, config_model_json)
        # produce parameters
        config['config_json'] = os.path.join(
            config['base_dir'],
            'config.json'
        )
        config['binary_path'] = os.path.join(
            config['base_dir'],
            'pytorch_model.bin'
        )
        # set parameters
        for param in self.deeplearning_params:
            self.init_param(config, *param)
        return

    def load(self: ConfigDeepLearning, config_model_json: str) -> Dict:
        config = dict()
        self.load_one(config, config_model_json)
        return config

    def save(self: ConfigDeepLearning, config: Dict) -> None:
        # save parameters
        for name, _, _, _ in self.deeplearning_params:
            self.save_param(config, name)
        # call parent function
        return super().save(config)


class DeepLearning(nn.Module):
    def __init__(self: DeepLearning) -> None:
        # call parent
        super().__init__()
        return
