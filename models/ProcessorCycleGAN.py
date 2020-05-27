#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from .Processor import ConfigProcessor, Processor
from .ImageLoaderCycleGAN import ImageLoaderCycleGAN
from .CycleGAN import CycleGAN
from .LightningCycleGAN import LightningCycleGAN


class ConfigProcessorCycleGAN(ConfigProcessor):
    def __init__(
        self: ConfigProcessorCycleGAN,
        config_json: str
    ) -> None:
        # load json
        config = dict()
        self._load_one(config, config_json)
        # set parameters
        for param in self.processor_params:
            self._init_param(config, *param)
        # value assertion
        assert(self.predict is False)
        # set classes
        self.prep_class = ImageLoaderCycleGAN
        self.model_class = CycleGAN
        self.light_class = LightningCycleGAN
        return


def convert_fake(fake: torch.Tensor, size: Tuple[int]) -> np.array:
    img = fake.cpu().squeeze(dim=0).detach()
    compose = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size)
    ])
    img = compose(img)
    return np.array(img, dtype=np.uint8)


class ProcessorCycleGAN(Processor):
    def __init__(
        self: ProcessorCycleGAN,
        config_json: str
    ) -> None:
        # call parent function
        self.config = ConfigProcessorCycleGAN(config_json)
        return

    def _cache_preprocess(self: ProcessorCycleGAN, preps: Dict) -> None:
        # call parent function
        super()._cache_preprocess(preps)
        prep = preps['prep']
        self.takeover_config = {
            'data_name': prep.config.data_name,
            'base_dir': prep.config.base_dir,
        }
        self.nameA = prep.config.name_a
        self.nameB = prep.config.name_b
        os.makedirs(self.output_prefix, exist_ok=True)
        os.makedirs(
            os.path.join(self.output_prefix, self.nameA),
            exist_ok=True
        )
        os.makedirs(
            os.path.join(self.output_prefix, self.nameB),
            exist_ok=True
        )
        return

    def ckpt_func(
        self: ProcessorCycleGAN,
        model: nn.Module,
        dataset: TensorDataset,
        epoch: int
    ) -> None:
        assert(dataset[0].size(0) == 1)
        model.eval()
        with torch.no_grad():
            ret = model(
                dataA=dataset[0].to(self.device),
                dataB=dataset[1].to(self.device)
            )
        fakeB = convert_fake(ret['fakeB'], (256, 256))
        fakeA = convert_fake(ret['fakeA'], (256, 256))
        fakeB.save(
            os.path.join(self.output_prefix, self.nameA, "%03d.png" % epoch)
        )
        fakeA.save(
            os.path.join(self.output_prefix, self.nameB, "%03d.png" % epoch)
        )
        model.train(True)
        return

    def _predict(
        self: ProcessorCycleGAN,
        dataloader: DataLoader,
        resources: List[Dict]
    ) -> None:
        for i, batch in enumerate(dataloader):
            assert(i == 0)
            assert(batch[0].size(0) == 1)
            with torch.no_grad():
                ret = self.model(
                    dataA=batch[0].to(self.device),
                    dataB=batch[1].to(self.device)
                )
            resources[0]['fakeB'] = convert_fake(ret['fakeB'], (256, 256))
            resources[0]['fakeA'] = convert_fake(ret['fakeA'], (256, 256))
        return

    def _output_resources(
        self: ProcessorCycleGAN,
        resources: List[Dict],
        output_fname: str
    ) -> None:
        res = resources[0]
        hnum = 2
        wnum = 2
        fig = plt.figure(figsize=(wnum * 5, hnum * 5), facecolo='w')
        for i, (img, title) in enumerate(zip(
            [res['rawA'], res['fakeB'], res['rawB'], res['fakeA']],
            [
                'real %s' % self.nameA,
                'fake %s' % self.nameB,
                'real %s' % self.nameB,
                'fake %s' % self.nameA,
            ]
        )):
            ax = fig.add_subplot(hnum, wnum, i + 1)
            ax.imshow(img)
            ax.set_axis_off()
            ax.set_title(title)
        plt.savefig(
            output_fname,
            facecolor='w', edgecolor='w',
            bbox_inches='tight', pad_inches=0.1
        )
        return
