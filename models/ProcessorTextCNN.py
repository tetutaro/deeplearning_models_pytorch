#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import DataLoader
from .Processor import ConfigProcessor, Processor
from .TokenizerTextCNN import TokenizerTextCNN
from .TextCNN import TextCNN
from .LightningTextCNN import LightningTextCNN


class ConfigProcessorTextCNN(ConfigProcessor):
    def __init__(self: ConfigProcessorTextCNN, config_json: str) -> None:
        # load json
        config = dict()
        self._load_one(config, config_json)
        # set parameters
        for param in self.processor_params:
            self._init_param(config, *param)
        # set classes
        self.prep_class = TokenizerTextCNN
        self.model_class = TextCNN
        self.light_class = LightningTextCNN
        return


class ProcessorTextCNN(Processor):
    def __init__(
        self: ProcessorTextCNN,
        config_json: str
    ) -> None:
        self.config = ConfigProcessorTextCNN(config_json)
        return

    def _cache_preprocess(self: ProcessorTextCNN, preps: Dict) -> None:
        # call parent function
        super()._cache_preprocess(preps)
        prep = preps['prep']
        self.takeover_config = {
            'data_name': prep.config.data_name,
            'base_dir': prep.config.base_dir,
            'num_class': prep.config.num_class,
            'word_len': prep.config.word_len,
            'word_vectors': prep.word_vectors,
        }
        self.unique_categories = prep.config.unique_categories
        return

    def _predict(
        self: ProcessorTextCNN,
        dataloader: DataLoader,
        resources: List[Dict]
    ) -> None:
        pred_cats = list()
        pred_probs = list()
        for batch in dataloader:
            with torch.no_grad():
                pred = self.model(batch[0].to(self.device))
            pred = pred.cpu().detach().numpy()
            pred_prob = np.max(pred, axis=1).tolist()
            pred_label = np.argmax(pred, axis=1).tolist()
            label2cat = self.unique_categories
            pred_cat = [label2cat[l] for l in pred_label]
            pred_cats.extend(pred_cat)
            pred_probs.extend(pred_prob)
        for res, cat, prob in zip(
            resources, pred_cats, pred_probs
        ):
            res['predicted_category'] = cat
            res['probability'] = "%.6f" % prob
        return

    def _output_resources(
        self: ProcessorTextCNN,
        resources: List[Dict],
        output_fname: str
    ) -> None:
        return self._output_json(resources, output_fname)
