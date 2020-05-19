#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple
from torch.utils.data import TensorDataset
from .ImageLoader import ConfigImageLoader, ImageLoader


class ConfigImageLoaderGradCAM(ConfigImageLoader):
    imageloader_gradcam_params = [
        # name, vtype, is_require, default
        ('model_name', str, True, None),
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
        return self.load_image()
