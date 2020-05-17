#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, Optional
import os
from abc import ABC
import simplejson as json


def _assert_type(val: Any, val_type: Any, offset: int = 0) -> None:
    if isinstance(val_type, list):
        if offset == len(val_type) - 1:
            return
        assert(isinstance(val, val_type[offset]))
        for v in val:
            _assert_type(v, val_type, offset=offset+1)
    else:
        assert(isinstance(val, val_type))
    return


class Config(ABC):
    def _init_param(
        self: Config,
        config: Dict,
        key: str,
        val_type: Any,
        is_require: bool,
        default: Optional[Any]
    ) -> None:
        val = config.get(key)
        if val is None:
            if is_require is True:
                print(key)
            assert(is_require is False)
            val = default
        if val is not None:
            _assert_type(val, val_type)
        setattr(self, key, val)
        return

    @staticmethod
    def _load_one(
        config: Dict,
        config_json_one: str,
    ) -> None:
        # open & read json files
        assert(os.path.exists(config_json_one))
        with open(config_json_one, 'rt') as rf:
            config_one = json.load(rf)
        config.update(config_one)
        return

    @staticmethod
    def _load_two(
        config: Dict,
        config_json_one: str,
        config_json_two: str
    ) -> None:
        # open & read json files
        assert(os.path.exists(config_json_one))
        with open(config_json_one, 'rt') as rf:
            config_one = json.load(rf)
        config.update(config_one)
        assert(os.path.exists(config_json_two))
        with open(config_json_two, 'rt') as rf:
            config_two = json.load(rf)
        config.update(config_two)
        return

    def _save(self: Config, config: Dict) -> None:
        config_json = getattr(self, 'config_json')
        with open(config_json, 'wt') as wf:
            json.dump(config, wf, ensure_ascii=False)
        return

    def _save_param(self: Config, config: Dict, key: str) -> None:
        config[key] = getattr(self, key)
        return
