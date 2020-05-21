#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import DataLoader
from .Processor import ConfigProcessor, Processor
from .TokenizerBertSum import TokenizerBertSum
from .BertSum import BertSum
from .LightningBertSum import LightningBertSum

DEFAULT_NUM_SELECT_SENTENCES = 3


class ConfigProcessorBertSum(ConfigProcessor):
    processor_bertsum_params = [
        ('num_select_sentences', int, False, DEFAULT_NUM_SELECT_SENTENCES),
    ]

    def __init__(
        self: ConfigProcessorBertSum,
        config_json: str
    ) -> None:
        # load json
        config = dict()
        self._load_one(config, config_json)
        # set parameters
        for param in self.processor_params:
            self._init_param(config, *param)
        for param in self.processor_bertsum_params:
            self._init_param(config, *param)
        # set classes
        self.prep_class = TokenizerBertSum
        self.model_class = BertSum
        self.light_class = LightningBertSum
        return


class ProcessorBertSum(Processor):
    def __init__(
        self: ProcessorBertSum,
        config_json: str
    ) -> None:
        self.config = ConfigProcessorBertSum(config_json)
        return

    def _cache_preprocess(
        self: ProcessorBertSum,
        preps: Dict
    ) -> None:
        # call parent function
        super()._cache_preprocess(preps)
        prep = preps['prep']
        self.takeover_config = {
            'data_name': prep.config.data_name,
            'base_dir': prep.config.base_dir,
            'nict_bert_dir': prep.config.nict_bert_dir,
        }
        return

    def _predict(
        self: ProcessorBertSum,
        dataloader: DataLoader,
        resources: List[Dict]
    ) -> None:
        sentences_list = [r['sentences'] for r in resources]
        predicted = list()
        offset = 0
        for batch in dataloader:
            batch_size = batch[0].size(0)
            batch_sentences = sentences_list[offset:offset+batch_size]
            with torch.no_grad():
                scores = self.model.forward(
                    input_ids=batch[0].to(self.device),
                    attention_mask=batch[1].to(self.device),
                    segment_ids=batch[2].to(self.device),
                    cls_ids=batch[3].to(self.device),
                    cls_ids_mask=batch[4].to(self.device),
                    cls_ids_mask_bool=batch[5].to(self.device)
                )
            cls_ids_mask = batch[4].cpu()
            scores = scores.cpu() + cls_ids_mask
            scores = scores.detach().numpy()
            selected_idxs = np.argsort(-scores, axis=1)
            for sents, idxs in zip(batch_sentences, selected_idxs):
                pred_abst = list()
                for idx in idxs:
                    if idx >= len(sents):
                        continue
                    pred_abst.append(sents[idx])
                    if len(pred_abst) == self.config.num_select_sentences:
                        break
                predicted.append(''.join(pred_abst))
            offset += batch_size
        for res, pred in zip(resources, predicted):
            res['predicted_abstruct'] = pred
            del res['sentences']
        return

    def _output_resources(
        self: ProcessorBertSum,
        resources: List[Dict],
        output_fname: str
    ) -> None:
        return self._output_json(resources, output_fname)
