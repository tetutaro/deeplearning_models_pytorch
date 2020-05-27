#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Callable
import os
from abc import ABC
from glob import glob
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import TensorDataset
from .Preprocessor import ConfigPreprocessor, Preprocessor


class ConfigImageLoader(ConfigPreprocessor, ABC):
    imageloader_params = [
        # name, vtype, is_require, default
        ('image_dir', str, True, None),
        ('extensions', [list, str], True, None),
    ]

    def _init_imageloader(
        self: ConfigImageLoader,
        config: dict,
        make_dir: bool
    ) -> None:
        # set parameters
        for param in self.imageloader_params:
            self._init_param(config, *param)
        if self.image_dir.endswith(os.sep):
            self.image_dir = self.image_dir[:-1]
        # value assertion
        assert(os.path.exists(self.image_dir))
        # internal parameters
        self.data_name = self.image_dir.split(os.sep)[-1]
        if make_dir:
            self.base_dir = os.path.join(
                'binaries', config['model_name'], self.data_name
            )
            os.makedirs(self.base_dir, exist_ok=True)
        return


class ImageDataset(TensorDataset):
    def __init__(
        self: ImageDataset,
        base_dir: str,
        image_dirs: Optional[List[str]],
        extensions: List[str],
        shuffle: bool,
        transform: Optional[Callable],
        preload: bool
    ) -> None:
        super().__init__()
        search_dirs = list()
        if image_dirs is None:
            search_dirs.append(base_dir)
        else:
            for d in image_dirs:
                search_dirs.append(os.path.join(base_dir, d))
        image_paths = list()
        for d in search_dirs:
            if d.endswith(tuple(['.' + x for x in extensions])):
                image_paths.append(d)
                continue
            for e in extensions:
                for image_path in glob(
                    os.path.join(d, '*.' + e)
                ):
                    image_paths.append(image_path)
        self.image_paths = sorted(image_paths)
        self.fnames = [
            os.path.splitext(
                os.path.basename(p)
            )[0] for p in self.image_paths
        ]
        self.shuffle = shuffle
        self.transform = transform
        self.preload = preload
        if preload:
            self.raws = list()
            self.imgs = list()
            for p in self.image_paths:
                raw = Image.open(p).convert('RGB')
                if self.transform is None:
                    img = transforms.ToTensor()(raw.copy())
                else:
                    img = self.transform(raw.copy())
                self.raws.append(np.array(raw, dtype=np.uint8)[..., ::-1])
                self.imgs.append(img)
        return

    def __getitem__(
        self: ImageDataset,
        index: int
    ) -> torch.Tensor:
        if self.shuffle:
            idx = np.random.randint(0, len(self.image_paths))
        else:
            idx = index % len(self.image_paths)
        if self.preload:
            return self.imgs[idx]
        image_path = self.image_paths[idx]
        raw = Image.open(image_path).convert('RGB')
        if self.transform is None:
            img = transforms.ToTensor()(raw.copy())
        else:
            img = self.transform(raw.copy())
        return img

    def __len__(self: ImageDataset) -> int:
        return len(self.image_paths)


class TwoImageDataset(TensorDataset):
    def __init__(
        self: TwoImageDataset,
        base_dir: str,
        image_dirs: List[List[str]],
        extensions: List[str],
        shuffles: List[bool],
        transform: Optional[Callable],
        preload: bool
    ) -> None:
        self.datasetA = ImageDataset(
            base_dir=base_dir,
            image_dirs=image_dirs[0],
            extensions=extensions,
            shuffle=shuffles[0],
            transform=transform,
            preload=preload
        )
        self.datasetB = ImageDataset(
            base_dir=base_dir,
            image_dirs=image_dirs[1],
            extensions=extensions,
            shuffle=shuffles[1],
            transform=transform,
            preload=preload
        )
        return

    def __getitem__(
        self: TwoImageDataset,
        index: int
    ) -> Tuple[torch.Tensor]:
        return (
            self.datasetA.__getitem__(index),
            self.datasetB.__getitem__(index)
        )

    def __len__(self: TwoImageDataset) -> int:
        return max(
            self.datasetA.__len__(),
            self.datasetB.__len__()
        )


class ImageLoader(Preprocessor, ABC):
    def load_image(
        self: ImageLoader,
        transform: Optional[Callable],
    ) -> Tuple[ImageDataset, List[Dict]]:
        dataset = ImageDataset(
            base_dir=self.config.image_dir,
            image_dirs=None,
            extensions=self.config.extensions,
            shuffle=False,
            transform=transform,
            preload=True
        )
        resources = [
            {"name": n, "raw": r} for n, r in zip(
                dataset.fnames, dataset.raws
            )
        ]
        return dataset, resources

    def create_ABdataset(
        self: ImageLoader,
        image_dirs: List[List[str]],
        shuffles: List[bool],
        transform: Optional[Callable],
        preload: bool
    ) -> Tuple[TwoImageDataset, List[Dict]]:
        ABdataset = TwoImageDataset(
            base_dir=self.config.image_dir,
            image_dirs=image_dirs,
            extensions=self.config.extensions,
            shuffles=shuffles,
            transform=transform,
            preload=preload
        )
        resources = list()
        for i in range(len(ABdataset)):
            tdic = dict()
            lenA = len(ABdataset.datasetA.fnames)
            lenB = len(ABdataset.datasetB.fnames)
            tdic['nameA'] = ABdataset.datasetA.fnames[i % lenA]
            tdic['nameB'] = ABdataset.datasetA.fnames[i % lenB]
            if preload:
                tdic['rawA'] = ABdataset.datasetA.raws[i % lenA]
                tdic['rawB'] = ABdataset.datasetB.raws[i % lenB]
            resources.append(tdic)
        return ABdataset, resources
