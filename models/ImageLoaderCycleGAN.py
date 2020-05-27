#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple
from torch.utils.data import TensorDataset
from torchvision import transforms
from .ImageLoader import ConfigImageLoader, ImageLoader


class ConfigImageLoaderCycleGAN(ConfigImageLoader):
    imageloader_cyclegan_params = [
        # name, vtype, is_require, default
        ('model_name', str, True, None),
        ('name_a', str, True, None),
        ('name_b', str, True, None),
        ('subdirs_a', [list, str], True, None),
        ('subdirs_b', [list, str], True, None),
        ('shuffle_a', bool, True, None),
        ('shuffle_b', bool, True, None),
        ('test_image', [list, str], True, None),
        ('preload', bool, True, None),
    ]

    def __init__(
        self: ConfigImageLoaderCycleGAN,
        config_data_json: str,
        config_preprocess_json: str
    ) -> None:
        config = dict()
        config['model_name'] = 'CycleGAN'
        self._load_two(config, config_data_json, config_preprocess_json)
        self._init_imageloader(config, make_dir=True)
        for param in self.imageloader_cyclegan_params:
            self._init_param(config, *param)
        return

    def load(self: ConfigImageLoader) -> None:
        return

    def save(self: ConfigImageLoader) -> None:
        return


class ImageLoaderCycleGAN(ImageLoader):
    def __init__(
        self: ImageLoaderCycleGAN,
        config_data_json: str,
        config_preprocess_json: str
    ) -> None:
        self.config = ConfigImageLoaderCycleGAN(
            config_data_json,
            config_preprocess_json
        )
        self.transform = transforms.Compose([
            transforms.Resize((286, 286)),
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])
        return

    def load(self: ImageLoader) -> None:
        return

    def save(self: ImageLoader) -> None:
        return

    def preprocess(
        self: ImageLoader
    ) -> Tuple[TensorDataset, List[Dict]]:
        return self.create_ABdataset(
            image_dirs=[self.config.subdirs_a, self.config.subdirs_b],
            shuffles=[self.config.shuffle_a, self.config.shuffle_b],
            transform=self.transform,
            preload=self.config.preload
        )

    def test_preprocess(
        self: ImageLoader
    ) -> Tuple[TensorDataset, List[Dict]]:
        return self.create_ABdataset(
            image_dirs=[
                [self.config.test_image[0]],
                [self.config.test_image[1]]
            ],
            shuffles=[False, False],
            transform=self.transform,
            preload=True
        )
