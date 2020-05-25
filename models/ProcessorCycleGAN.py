#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List
from torch.utils.data import DataLoader
from .Processor import ConfigProcessor, Processor
from .ImageLoaderCycleGAN import ImageLoaderCycleGAN
from .CycleGAN import CycleGAN
from .LightningCycleGAN import LightningCycleGAN


class ConfigProcessorCycleGAN(ConfigProcessor):
    def __init__(
        self: ConfigProcessorCycleGAN,
        config_json: str
    ) -> None:
        # load json
        config = dict()
        self._load_one(config, config_json)
        # set parameters
        for param in self.processor_params:
            self._init_param(config, *param)
        # value assertion
        assert(self.predict is False)
        # set classes
        self.prep_class = ImageLoaderCycleGAN
        self.model_class = CycleGAN
        self.light_class = LightningCycleGAN
        return


class ProcessorCycleGAN(Processor):
    def __init__(
        self: ProcessorCycleGAN,
        config_json: str
    ) -> None:
        # call parent function
        self.config = ConfigProcessorCycleGAN(config_json)
        return

    def _cache_preprocess(self: ProcessorCycleGAN, preps: Dict) -> None:
        # call parent function
        super()._cache_preprocess(preps)
        prep = preps['prep']
        self.takeover_config = {
            'data_name': prep.config.data_name,
            'base_dir': prep.config.base_dir,
        }
        return

    def _predict(
        self: ProcessorCycleGAN,
        dataloader: DataLoader,
        resources: List[Dict]
    ) -> None:
        return

    def _output_resources(
        self: ProcessorCycleGAN,
        resources: List[Dict],
        output_fname: str
    ) -> None:
        return
