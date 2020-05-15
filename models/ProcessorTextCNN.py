#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List
import simplejson as json
import numpy as np
import torch
from torch.utils.data import DataLoader
from .Processor import ConfigProcessor, Processor
from .TokenizerTextCNN import ConfigTokenizerTextCNN, TokenizerTextCNN
from .TextCNN import ConfigTextCNN, TextCNN
from .LightningTextCNN import ConfigLightningTextCNN, LightningTextCNN


class ConfigProcessorTextCNN(ConfigProcessor):
    def __init__(self: ConfigProcessorTextCNN, config_json: str) -> None:
        super().__init__(config_json)
        self.config_prep_class = ConfigTokenizerTextCNN
        self.prep_class = TokenizerTextCNN
        self.config_model_class = ConfigTextCNN
        self.model_class = TextCNN
        self.config_light_class = ConfigLightningTextCNN
        self.light_class = LightningTextCNN
        return


class ProcessorTextCNN(Processor):
    def __init__(
        self: ProcessorTextCNN,
        config: ConfigProcessorTextCNN
    ) -> None:
        # call parent function
        super().__init__(config)
        self.config = config
        return

    def cache_preprocess(self: ProcessorTextCNN, preps: Dict) -> None:
        # call parent function
        super().cache_preprocess(preps)
        self.takeover_config = {
            'base_dir': preps['config_prep'].base_dir,
            'num_class': preps['config_prep'].num_class,
            'word_len': preps['config_prep'].word_len,
            'word_vectors': preps['prep'].word_vectors,
        }
        self.unique_categories = preps['config_prep'].unique_categories
        self.category_column = preps['config_prep'].category_column
        return

    def predict(
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
        category_column = self.category_column
        for res, cat, prob in zip(
            resources, pred_cats, pred_probs
        ):
            res['predicted_%s' % category_column] = cat
            res['probability'] = "%.6f" % prob
        return

    def output_resources(
        self: Processor,
        resources: List[Dict],
        dtype: str
    ) -> None:
        output_fname = self.get_output_fname(dtype)
        with open(output_fname, 'wt') as wf:
            forward_str = "[\n"
            for res in resources:
                wf.write(forward_str)
                json.dump(res, wf, ensure_ascii=False)
                forward_str = ",\n"
            wf.write("\n]\n")
        return
