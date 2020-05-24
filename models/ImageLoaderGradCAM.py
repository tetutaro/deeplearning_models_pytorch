#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple
import os
from torch.utils.data import TensorDataset
from torchvision import transforms
from .ImageLoader import ConfigImageLoader, ImageLoader


class ConfigImageLoaderGradCAM(ConfigImageLoader):
    imageloader_gradcam_params = [
        # name, vtype, is_require, default
        ('model_name', str, True, None),
        ('categories_path', str, True, None),
    ]

    def __init__(
        self: ConfigImageLoaderGradCAM,
        config_data_json: str,
        config_preprocess_json: str
    ) -> None:
        config = dict()
        config['model_name'] = 'GradCAM'
        self._load_two(config, config_data_json, config_preprocess_json)
        self._init_imageloader(config)
        for param in self.imageloader_gradcam_params:
            self._init_param(config, *param)
        # value assertion
        assert(os.path.exists(self.categories_path))
        # internal parameters
        unique_categories = list()
        with open(self.categories_path, 'rt') as rf:
            line = rf.readline()
            while line:
                unique_categories.append(line.strip())
                line = rf.readline()
        self.unique_categories = unique_categories
        self.num_class = len(unique_categories)
        return

    def load(self: ConfigImageLoader) -> None:
        return

    def save(self: ConfigImageLoader) -> None:
        return


class ImageLoaderGradCAM(ImageLoader):
    def __init__(
        self: ImageLoaderGradCAM,
        config_data_json: str,
        config_preprocess_json: str
    ) -> None:
        self.config = ConfigImageLoaderGradCAM(
            config_data_json,
            config_preprocess_json
        )
        return

    def load(self: ImageLoader) -> None:
        return

    def save(self: ImageLoader) -> None:
        return

    def preprocess(
        self: ImageLoader
    ) -> Tuple[TensorDataset, List[Dict]]:
        return self.load_image(
            transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        )
