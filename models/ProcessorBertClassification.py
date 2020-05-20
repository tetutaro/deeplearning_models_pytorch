#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .Processor import ConfigProcessor, Processor
from .TokenizerBertClassification import TokenizerBertClassification
from .BertClassification import BertClassification
from .LightningBertClassification import LightningBertClassification


class ConfigProcessorBertClassification(ConfigProcessor):
    def __init__(
        self: ConfigProcessorBertClassification,
        config_json: str
    ) -> None:
        # load json
        config = dict()
        self._load_one(config, config_json)
        # set parameters
        for param in self.processor_params:
            self._init_param(config, *param)
        # set classes
        self.prep_class = TokenizerBertClassification
        self.model_class = BertClassification
        self.light_class = LightningBertClassification
        return


class ProcessorBertClassification(Processor):
    def __init__(
        self: ProcessorBertClassification,
        config_json: str
    ) -> None:
        self.config = ConfigProcessorBertClassification(config_json)
        return

    def _cache_preprocess(
        self: ProcessorBertClassification,
        preps: Dict
    ) -> None:
        # call parent function
        super()._cache_preprocess(preps)
        prep = preps['prep']
        self.takeover_config = {
            'data_name': prep.config.data_name,
            'base_dir': prep.config.base_dir,
            'nict_bert_dir': prep.config.nict_bert_dir,
            'num_class': prep.config.num_class,
            'unique_categories': prep.config.unique_categories,
        }
        self.unique_categories = prep.config.unique_categories
        return

    def _predict(
        self: ProcessorBertClassification,
        dataloader: DataLoader,
        resources: List[Dict]
    ) -> None:
        pred_cats = list()
        pred_probs = list()
        for batch in dataloader:
            with torch.no_grad():
                _, logit = self.model.forward((
                    batch[0].to(self.device),
                    batch[1].to(self.device),
                    batch[2].to(self.device),
                    batch[3].to(self.device)
                ))
            probs = F.softmax(logit, dim=1).cpu().detach().numpy()
            label2cat = self.unique_categories
            pred_prob = np.max(probs, axis=1).tolist()
            pred_label = np.argmax(probs, axis=1).tolist()
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
        self: ProcessorBertClassification,
        resources: List[Dict],
        output_fname: str
    ) -> None:
        return self._output_json(resources, output_fname)
