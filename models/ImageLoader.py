#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple
import os
from abc import ABC
from glob import glob
import cv2
import torch
from torchvision import transforms
from torch.utils.data import TensorDataset
from .Preprocessor import ConfigPreprocessor, Preprocessor


class ConfigImageLoader(ConfigPreprocessor, ABC):
    imageloader_params = [
        # name, vtype, is_require, default
        ('image_dir', str, True, None),
        ('categories_path', str, True, None),
        ('num_class', int, False, 0),
        ('unique_categories', [list, str], False, []),
    ]

    def _init_imageloader(
        self: ConfigImageLoader,
        config: dict
    ) -> None:
        # set parameters
        for param in self.imageloader_params:
            self._init_param(config, *param)
        if self.image_dir.endswith(os.sep):
            self.image_dir = self.image_dir[:-1]
        # value assertion
        assert(os.path.exists(self.image_dir))
        assert(os.path.exists(self.categories_path))
        # internal parameters
        self.data_name = self.image_dir.split(os.sep)[-1]
        unique_categories = list()
        with open(self.categories_path, 'rt') as rf:
            line = rf.readline()
            while line:
                unique_categories.append(line.strip())
                line = rf.readline()
        self.unique_categories = unique_categories
        self.num_class = len(unique_categories)
        return


class ImageLoader(Preprocessor, ABC):
    def load_image(
        self: ImageLoader
    ) -> Tuple[TensorDataset, List[Dict]]:
        resources = list()
        images = list()
        for fname in glob(
            os.path.join(self.config.image_dir, "*.png")
        ):
            raw = cv2.imread(fname)
            raw = cv2.resize(raw, (224,) * 2)
            img = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])(raw[..., ::-1].copy())
            images.append(torch.unsqueeze(img, 0))
            resources.append({
                "name": os.path.splitext(
                    os.path.basename(fname)
                )[0],
                "raw": raw,
            })
        cated_images = torch.cat(images, dim=0)
        return TensorDataset(cated_images), resources
