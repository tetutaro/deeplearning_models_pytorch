#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Generator
import os
from abc import ABC
import simplejson as json
from .Config import Config


class ConfigPreprocessor(Config, ABC):
    preprocessor_params = [
        # name, vtype, is_require, default
        ('data_json', str, True, None),
        ('data_name', str, True, None),
        ('base_dir', str, True, None),
        ('config_json', str, True, None),
    ]

    def _init_preprocessor(
        self: ConfigPreprocessor,
        config: Dict
    ) -> None:
        assert('model_name' in list(config.keys()))
        assert('data_json' in list(config.keys()))
        # produce parameters
        config['data_name'] = os.path.splitext(
            os.path.basename(config['data_json'])
        )[0]
        config['base_dir'] = os.path.join(
            'binaries', config['model_name'], config['data_name']
        )
        os.makedirs(config['base_dir'], exist_ok=True)
        config['config_json'] = os.path.join(
            config['base_dir'],
            'tokenizer_config.json'
        )
        return


class Preprocessor(ABC):
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
