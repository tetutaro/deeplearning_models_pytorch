#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Generator
import os
import simplejson as json
from .Config import Config


class ConfigPreprocessor(Config):
    preprocessor_params = [
        # name, vtype, is_require, default
        ('data_name', str, True, None),
        ('data_json', str, True, None),
        ('base_dir', str, True, None),
        ('config_json', str, True, None),
    ]

    def __init__(
        self: ConfigPreprocessor,
        config: Dict,
        config_data_json: str,
        config_preprocess_json: str
    ) -> None:
        # call parent function
        super().__init__()
        # load json
        self.load_two(
            config,
            config_data_json,
            config_preprocess_json
        )
        # produce parameters
        data_name = os.path.splitext(
            os.path.basename(config['data_json'])
        )[0]
        base_dir = os.path.join(
            'binaries', config['model_name'], data_name
        )
        os.makedirs(base_dir, exist_ok=True)
        config['config_json'] = os.path.join(
            base_dir,
            'tokenizer_config.json'
        )
        config['data_name'] = data_name
        config['base_dir'] = base_dir
        # set parameters
        for param in self.preprocessor_params:
            self.init_param(config, *param)
        return

    def load(self: ConfigPreprocessor, config_json: str) -> Dict:
        config = dict()
        # load json
        self.load_one(config, config_json)
        return config

    def save(self, config: Dict) -> None:
        # save parameters
        for name, _, _, _ in self.preprocessor_params:
            self.save_param(config, name)
        # call parent function
        return super().save(config)


class Preprocessor(object):
    def __init__(self, config: ConfigPreprocessor) -> None:
        self.config = config
        return

    def yield_data_json(
        self: Preprocessor
    ) -> Generator[Dict, None, None]:
        with open(self.config.data_json, 'rt') as rf:
            line = rf.readline()
            while line:
                line = line.strip()
                if line.endswith((',', '[', ']')):
                    line = line[:-1]
                if len(line) == 0:
                    line = rf.readline()
                    continue
                yield json.loads(line)
                line = rf.readline()
        return
