#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from .DeepLearning import ConfigDeepLearning

INPUT_CHANNELS = ['single', 'multi']
DEFAULT_INPUT_CHANNEL = 'single'
DEFAULT_DROP_PROB = 0.5
DEFAULT_RANDOM_STATE = None


class ConfigTextCNN(ConfigDeepLearning):
    textcnn_params = [
        # name, vtype, is_require, default
        ('num_class', int, True, None),
        ('word_len', int, True, None),
        ('padding_idx', int, False, None),
        ('kernels', [list, list, int], True, None),
        ('linear_dim', int, True, None),
        ('input_channel', str, False, DEFAULT_INPUT_CHANNEL),
        ('drop_prob', float, False, DEFAULT_DROP_PROB),
        ('random_state', int, False, DEFAULT_RANDOM_STATE),
    ]

    def __init__(
        self: ConfigTextCNN,
        config: Dict,
        config_model_json: str
    ) -> None:
        # model_name
        config['model_name'] = 'TextCNN'
        # load json
        self._load_one(config, config_model_json)
        # init parent class
        self._init_deeplearning(config)
        # produce parameters
        config['linear_dim'] = sum([o for f, o in config['kernels']])
        # set parameters
        for param in self.deeplearning_params:
            self._init_param(config, *param)
        for param in self.textcnn_params:
            self._init_param(config, *param)
        # set word_vectors
        setattr(self, 'word_vectors', config['word_vectors'])
        # value_assertion
        for k in self.kernels:
            assert(len(k) == 2)
            for d in k:
                assert(d > 0)
        assert(self.input_channel in INPUT_CHANNELS)
        assert(self.linear_dim > self.num_class)
        assert((self.drop_prob >= 0) and (self.drop_prob < 1))
        return

    def load(self: ConfigTextCNN) -> None:
        # load json
        config = dict()
        self._load_one(config, self.config_json)
        # set parameters
        for param in self.deeplearning_params:
            self._init_param(config, *param)
        for param in self.textcnn_params:
            self._init_param(config, *param)
        return

    def save(self: ConfigTextCNN) -> None:
        config = dict()
        # save parameters
        for name, _, _, _ in self.deeplearning_params:
            self._save_param(config, name)
        for name, _, _, _ in self.textcnn_params:
            self._save_param(config, name)
        self._save(config)
        return


class TextCNN(nn.Module):
    def __init__(
        self: TextCNN,
        config: Dict,
        config_model_json: str
    ) -> None:
        super().__init__()
        self.config = ConfigTextCNN(config, config_model_json)
        # set random seed
        if self.config.random_state is not None:
            np.random.seed(self.config.random_state)
            torch.manual_seed(self.config.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.config.random_state)
        # Embedding Layer
        in_channels = 1
        vocab_size = self.config.word_vectors.shape[0]
        word_dim = self.config.word_vectors.shape[1]
        embeds = list()
        embed = nn.Embedding(
            vocab_size, word_dim, padding_idx=0
        )
        embed.weight = nn.Parameter(
            torch.from_numpy(self.config.word_vectors)
        )
        embed.weight.requires_grad_(False)
        embeds.append(embed)
        if self.config.input_channel == 'multi':
            in_channels += 1
            embed = nn.Embedding(
                vocab_size, word_dim, padding_idx=0
            )
            embed.weight = nn.Parameter(
                torch.from_numpy(self.config.word_vectors)
            )
            embeds.append(embed)
        self.embeds = nn.ModuleList(embeds)
        # Normalization Layer
        self.norm = nn.LayerNorm(word_dim)
        # Convolutional Layer
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels, o, (f, word_dim)
            ) for f, o in self.config.kernels
        ])
        # Max Pooling Layer
        self.pools = nn.ModuleList([
            nn.MaxPool2d(
                (self.config.word_len - f + 1, 1)
            ) for f, o in self.config.kernels
        ])
        # Flatten Layer
        self.flat = nn.Flatten()
        # DropOut Layer
        self.drop = nn.Dropout(p=self.config.drop_prob)
        # LinearCombination Layer
        self.line = nn.Linear(
            self.config.linear_dim,
            self.config.num_class
        )
        # ReLU Layer
        self.relu = nn.ReLU()
        # Softmax Layer
        self.smax = nn.Softmax(dim=1)
        return

    def forward(self: TextCNN, x: torch.Tensor) -> torch.Tensor:
        # Embedding & Normalize
        x = self.norm(torch.cat([
            embed(x).unsqueeze(1) for embed in self.embeds
        ], dim=1).float())
        # Convolution
        xs = [conv(x) for conv in self.convs]
        # MaxPooling & Flatten
        x = self.flat(
            torch.cat([pool(x) for pool, x in zip(self.pools, xs)], dim=1)
        )
        # Dense network
        return self.smax(self.relu(self.line(self.drop(x))))

    def load(self: TextCNN) -> None:
        self.config.load()
        if torch.cuda.is_available():
            map_location = torch.device('cuda')
        else:
            map_location = torch.device('cpu')
        self.load_state_dict(torch.load(
            self.config.binary_path, map_location=map_location
        ))
        return

    def save(self: TextCNN) -> None:
        self.config.save()
        torch.save(self.state_dict(), self.config.binary_path)
        return
