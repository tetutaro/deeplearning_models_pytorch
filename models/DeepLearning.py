#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict
import os
from abc import ABC
from .Config import Config


class ConfigDeepLearning(Config, ABC):
    deeplearning_params = [
        # name, vtype, is_require, default
        ('data_name', str, True, None),
        ('base_dir', str, True, None),
        ('config_json', str, True, None),
        ('binary_path', str, True, None),
    ]

    def _init_deeplearning(
        self: ConfigDeepLearning,
        config: Dict,
    ) -> None:
        assert('model_name' in list(config.keys()))
        assert('data_name' in list(config.keys()))
        assert('base_dir' in list(config.keys()))
        # produce parameters
        config['config_json'] = os.path.join(
            config['base_dir'],
            'config.json'
        )
        config['binary_path'] = os.path.join(
            config['base_dir'],
            'pytorch_model.bin'
        )
        return
