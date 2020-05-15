#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List
import os
from operator import itemgetter
from abc import abstractmethod
from sklearn.model_selection import ShuffleSplit
import torch
from torch.utils.data import DataLoader, Subset
from .Config import Config

DEFAULT_FIT = True
DEFAULT_REFIT = False
DEFAULT_PREDICT = True
DEFAULT_OUTPUT_TRAIN = False
DEFAULT_TEST_RATE = 0.1


class ConfigProcessor(Config):
    processor_params = [
        # name, vtype, is_require, default
        ('config_data_json', str, True, None),
        ('config_preprocessor_json', str, True, None),
        ('config_model_json', str, True, None),
        ('config_lightning_json', str, True, None),
        ('fit', bool, False, DEFAULT_FIT),
        ('refit', bool, False, DEFAULT_REFIT),
        ('predict', bool, False, DEFAULT_PREDICT),
        ('output_train', bool, False, DEFAULT_OUTPUT_TRAIN),
        ('test_rate', float, False, DEFAULT_TEST_RATE),
    ]

    def __init__(
        self: ConfigProcessor,
        config_json: str
    ) -> None:
        # call parent function
        super().__init__()
        config = dict()
        self.load_one(config, config_json)
        # set parameters
        for param in self.processor_params:
            self.init_param(config, *param)
        return


class Processor(object):
    def __init__(self: Processor, config: ConfigProcessor) -> None:
        self.config = config
        return

    def preprocess(self: Processor, load: bool) -> None:
        # setup config preprocessor
        config_prep = self.config.config_prep_class(
            self.config.config_data_json,
            self.config.config_preprocessor_json
        )
        if load:
            # load config preprocessor
            assert(os.path.exists(config_prep.config_json))
            config_prep.load(
                config_prep.config_json
            )
        # create proprocessor instance
        prep = self.config.prep_class(config_prep)
        # preprocess
        self.dataset, self.resources = prep.preprocess()
        if self.config.fit:
            prep.save()
        self.cache_preprocess({
            'config_prep': config_prep,
            'prep': prep,
        })
        return

    def cache_preprocess(self: Processor, preps: Dict) -> None:
        self.output_json_prefix = os.path.join(
            'results',
            "_".join([
                preps['config_prep'].model_name,
                preps['config_prep'].data_name
            ])
        )
        self.takeover_config = dict()
        return

    def split_data(self: Processor) -> None:
        # devide data into train and test
        if self.config.test_rate > 0:
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

    def fit(self: Processor, load: bool) -> None:
        config_model = self.config.config_model_class(
            self.takeover_config,
            self.config.config_model_json
        )
        config_light = self.config.config_light_class(
            self.config.config_lightning_json
        )
        # restore information to using them in followings
        self.saved_config_json = config_model.config_json
        self.batch_size = config_light.batch_size
        if not self.config.fit:
            return
        if load:
            assert(os.path.exists(self.saved_config_json))
            config_model.load(
                self.saved_config_json
            )
        model = self.config.model_class(config_model)
        if load:
            # load model
            model.load()
        # create lightning instance
        light = self.config.light_class(
            config_light, model, self.train_dataset
        )
        # fit
        light.fit()
        # save
        model.save()
        return

    def reload_model(self: Processor) -> None:
        # load model
        config_model = self.config.config_model_class(
            self.takeover_config,
            self.config.config_model_json
        )
        config_model.load(
            self.saved_config_json
        )
        self.model = self.config.model_class(config_model)
        self.model.load()
        return

    @abstractmethod
    def predict(
        self: Processor,
        dataloader: DataLoader,
        resources: List[Dict]
    ) -> None:
        for batch in dataloader:
            with torch.no_grad():
                pred = self.model(batch[0].to(self.device))
            _ = pred.cpu().detach().numpy()
        return

    def get_output_fname(self: Processor, dtype: str) -> str:
        if dtype == "train":
            postfix = "_train.json"
        elif dtype == "test":
            postfix = "_test.json"
        else:
            postfix = ".json"
        return self.output_json_prefix + postfix

    @abstractmethod
    def output_resources(
        self: Processor,
        resources: List[Dict],
        dtype: str
    ) -> None:
        output_fname = self.get_output_fname(dtype)
        with open(output_fname, 'wt') as wf:
            wf.write('')
        return

    def predict_and_output(self: Processor, dtype: str) -> None:
        # detect computing device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)
        # select data
        if dtype == "train":
            dataset = self.train_dataset
            resources = self.train_resources
        elif dtype == "test":
            dataset = self.test_dataset
            resources = self.test_resources
        else:
            dataset = self.dataset
            resources = self.resources
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size, shuffle=False, num_workers=0
        )
        self.predict(dataloader, resources)
        self.output_resources(resources, dtype)
        return

    def process(self: Processor):
        if self.config.fit:
            if self.config.refit:
                load = True
            else:
                load = False
        else:
            load = True
        self.preprocess(load)
        self.split_data()
        self.fit(load)
        if not self.config.predict:
            return
        self.reload_model()
        if self.config.fit:
            if self.config.output_train:
                self.predict_and_output("train")
            self.predict_and_output("test")
        else:
            self.predict_and_output("all")
        return
