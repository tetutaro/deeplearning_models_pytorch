#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
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
        return

    def _predict(
        self: ProcessorCycleGAN,
        dataloader: DataLoader,
        resources: List[Dict]
    ) -> None:
        if self.config.mode == 'image':
            self._predict_image(dataloader, resources)
        else:  # self.config.mode == 'text'
            self._predict_text(dataloader, resources)
        return

    @staticmethod
    def _mask_gcam(raw: np.array, gcam: np.array) -> np.array:
        cmap = cm.jet_r(gcam)[..., :3] * 255.0
        masked = (cmap.astype(np.float) + raw.astype(np.float)) * 0.5
        return np.uint8(masked)

    def _predict_image(
        self: ProcessorCycleGAN,
        dataloader: DataLoader,
        resources: List[Dict]
    ) -> None:
        category2label = {x: i for i, x in enumerate(self.unique_categories)}
        results = list()
        self.output_tags = ['raw'] + [
            x if x is not None else 'top1'
            for x in self.config.target_categories
        ]
        title_format = "predict {pred} from\n{name} (prob: {prob:.4f})"
        offset = 0
        for batch in dataloader:
            batch_size = batch.size(0)
            batch_results = list()
            for x in batch.clone().cpu().detach().numpy():
                rdic = resources[offset]
                batch_results.append({
                    'name': rdic['name'],
                    'raw': {
                        'title': "raw image %s" % rdic['name'],
                        'image': rdic['raw']
                    }
                })
                del resources[offset]['raw']
                offset += 1
            for xcat in self.config.target_categories:
                if xcat is None:
                    xpred = None
                    ytag = 'top1'
                else:
                    xpred = np.array([category2label[xcat]] * batch_size)
                    ytag = xcat
                res = self.model(batch.to(self.device), pred=xpred)
                for bres, gcam, prob, pred in zip(
                    batch_results, res['gcam'], res['prob'], res['pred']
                ):
                    pred = self.unique_categories[pred]
                    raw = bres['raw']['image']
                    name = bres['name']
                    bres[ytag] = {
                        'image': self._mask_gcam(raw, gcam),
                        'title': title_format.format(
                            name=name, pred=pred, prob=prob
                        ),
                    }
            results.extend(batch_results)
        for resource, result in zip(resources, results):
            resource.update(result)
        return

    def _output_resources(
        self: ProcessorCycleGAN,
        resources: List[Dict],
        output_fname: str
    ) -> None:
        self._output_image(resources, output_fname)
        return

    def _output_image(
        self: ProcessorCycleGAN,
        resources: List[Dict],
        output_fname: str
    ) -> None:
        hnum = len(self.output_tags)
        wnum = len(resources)
        fig = plt.figure(figsize=(wnum * 5, hnum * 5), facecolor='w')
        off = 1
        for tag in self.output_tags:
            for res in resources:
                ax = fig.add_subplot(hnum, wnum, off)
                tdic = res[tag]
                ax.imshow(tdic['image'][..., ::-1])
                ax.set_axis_off()
                ax.set_title(tdic['title'])
                off += 1
        plt.savefig(
            output_fname,
            facecolor='w', edgecolor='w',
            bbox_inches='tight', pad_inches=0.1
        )
        return
