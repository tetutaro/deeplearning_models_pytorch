#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List
import os
from operator import itemgetter
from abc import ABC, abstractmethod
import simplejson as json
from sklearn.model_selection import ShuffleSplit
import torch
from torch.utils.data import DataLoader, Subset
from .Config import Config

DEFAULT_BATCH_SIZE = 100
DEFAULT_FIT = True
DEFAULT_REFIT = False
DEFAULT_PREDICT = True
DEFAULT_OUTPUT_TRAIN = False
DEFAULT_TEST_RATE = 0.1


class ConfigProcessor(Config, ABC):
    processor_params = [
        # name, vtype, is_require, default
        ('config_data_json', str, True, None),
        ('config_preprocessor_json', str, True, None),
        ('config_model_json', str, True, None),
        ('config_lightning_json', str, True, None),
        ('batch_size', int, False, DEFAULT_BATCH_SIZE),
        ('test_rate', float, False, DEFAULT_TEST_RATE),
        ('fit', bool, False, DEFAULT_FIT),
        ('refit', bool, False, DEFAULT_REFIT),
        ('predict', bool, False, DEFAULT_PREDICT),
        ('output_extension', str, True, None),
        ('output_train', bool, False, DEFAULT_OUTPUT_TRAIN),
    ]


class Processor(ABC):
    def _preprocess(self: Processor, load: bool) -> None:
        # create preprocessor instance
        prep = self.config.prep_class(
            self.config.config_data_json,
            self.config.config_preprocessor_json
        )
        if load:
            prep.load()
        # preprocess
        self.dataset, self.resources = prep.preprocess()
        if hasattr(prep, 'test_preprocess'):
            self.test_dataset, self.test_resources = prep.test_preprocess()
        if not load:
            prep.save()
        self._cache_preprocess({'prep': prep})
        return

    @abstractmethod
    def _cache_preprocess(self: Processor, preps: Dict) -> None:
        prep = preps['prep']
        self.output_prefix = os.path.join(
            'results',
            "_".join([
                prep.config.model_name,
                prep.config.data_name
            ])
        )
        self.takeover_config = dict()
        return

    def _split_data(self: Processor) -> None:
        # devide data into train and test
        if self.config.test_rate > 0:
            assert(hasattr(self, 'test_dataset') is False)
            assert(hasattr(self, 'test_resources') is False)
            ss = ShuffleSplit(n_splits=1, test_size=self.config.test_rate)
            train_idx, test_idx = list(ss.split(self.resources))[0]
            self.train_dataset = Subset(self.dataset, train_idx)
            self.test_dataset = Subset(self.dataset, test_idx)
            self.train_resources = itemgetter(*train_idx)(self.resources)
            self.test_resources = itemgetter(*test_idx)(self.resources)
        else:
            self.train_dataset = self.dataset
            self.train_resources = self.resources
        return

    def _fit(self: Processor, load: bool) -> None:
        # create model instance
        model = self.config.model_class(
            self.takeover_config,
            self.config.config_model_json
        )
        if load:
            model.load()
        # create lightning instance
        if hasattr(self.ckpt_func):
            light = self.config.light_class(
                self.config.config_lightning_json,
                model,
                self.train_dataset,
                self.ckpt_func,
                self.test_dataset
            )
        else:
            light = self.config.light_class(
                self.config.config_lightning_json,
                model,
                self.train_dataset
            )
        # fit
        light.fit()
        # save
        model.save()
        return

    def _reload_model(self: Processor) -> None:
        # load model
        self.model = self.config.model_class(
            self.takeover_config,
            self.config.config_model_json
        )
        self.model.load()
        self.model.eval()
        return

    @abstractmethod
    def _predict(
        self: Processor,
        dataloader: DataLoader,
        resources: List[Dict]
    ) -> None:
        for batch in dataloader:
            with torch.no_grad():
                pred = self.model(batch[0].to(self.device))
            _ = pred.cpu().detach().numpy()
        return

    def _get_output_fname(self: Processor, dtype: str) -> str:
        fname = self.output_prefix
        if dtype == "train":
            fname += "_train." + self.config.output_extension
        elif dtype == "test":
            fname += "_test." + self.config.output_extension
        else:
            fname += "." + self.config.output_extension
        return fname

    @abstractmethod
    def _output_resources(
        self: Processor,
        resources: List[Dict],
        output_fname: str
    ) -> None:
        with open(output_fname, 'wt') as wf:
            wf.write('')
        return

    def _output_json(
        self: Processor,
        resources: List[Dict],
        output_fname: str
    ) -> None:
        with open(output_fname, 'wt') as wf:
            forward_str = "[\n"
            for res in resources:
                wf.write(forward_str)
                json.dump(res, wf, ensure_ascii=False)
                forward_str = ",\n"
            wf.write("\n]\n")
        return

    def _predict_and_output(self: Processor, dtype: str) -> None:
        self.model.to(self.device)
        # select data
        if dtype == "train":
            dataset = self.train_dataset
            resources = self.train_resources
        elif dtype == "test":
            dataset = self.test_dataset
            resources = self.test_resources
        else:
            if hasattr(self, 'test_dataset'):
                dataset = self.test_dataset
                resources = self.test_resources
            else:
                dataset = self.dataset
                resources = self.resources
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size, shuffle=False, num_workers=0
        )
        self._predict(dataloader, resources)
        output_fname = self._get_output_fname(dtype)
        self._output_resources(resources, output_fname)
        return

    def process(self: Processor):
        # detect computing device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        # create load flag
        if self.config.fit:
            if self.config.refit:
                load = True
            else:
                load = False
        else:
            load = True
        self._preprocess(load)
        self._split_data()
        if self.config.fit:
            self._fit(load)
        if not self.config.predict:
            return
        self._reload_model()
        if self.config.fit and self.config.test_rate > 0:
            if self.config.output_train:
                self._predict_and_output("train")
            self._predict_and_output("test")
        else:
            self._predict_and_output("all")
        return
