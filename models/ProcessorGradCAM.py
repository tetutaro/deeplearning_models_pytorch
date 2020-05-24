#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List
from html.parser import HTMLParser
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from .Processor import ConfigProcessor, Processor
from .ImageLoaderGradCAM import ImageLoaderGradCAM
from .TokenizerGradCAM import TokenizerGradCAM
from .GradCAM import GRADCAM_MODES, GradCAM

WHITE_COEF = 0.3


class ConfigProcessorGradCAM(ConfigProcessor):
    processor_gradcam_params = [
        # name, vtype, is_require, default
        ('mode', str, True, None),
        ('target_categories', list, True, None),
    ]

    def __init__(
        self: ConfigProcessorGradCAM,
        config_json: str
    ) -> None:
        # load json
        config = dict()
        self._load_one(config, config_json)
        # set parameters
        for param in self.processor_params:
            self._init_param(config, *param)
        for param in self.processor_gradcam_params:
            self._init_param(config, *param)
        # value assertion
        assert(self.mode in GRADCAM_MODES)
        assert(self.fit is False)
        assert(self.refit is False)
        # set classes
        if self.mode == 'image':
            self.prep_class = ImageLoaderGradCAM
        else:
            self.prep_class = TokenizerGradCAM
        self.model_class = GradCAM
        return


def make_bg(x: float) -> str:
    c = (
        (1 - ((1 - np.array(cm.jet_r(x)[:3])) * WHITE_COEF)) * 255
    ).astype(np.int)
    c[0], c[2] = c[2], c[0]  # BGR2RGB
    return ''.join([f"{x:02x}" for x in c])


class EmbedExplanation(HTMLParser):
    def __init__(self: EmbedExplanation, importances: np.array) -> None:
        super(EmbedExplanation, self).__init__()
        self.importances = importances
        self.off = 0
        self.embedded = ''
        return

    def handle_starttag(self: EmbedExplanation, tag: str, attrs: List) -> None:
        assert(tag == 'span')
        self.embedded += '<a class="popuped" href="#">'
        imp = self.importances[self.off]
        bg = make_bg(imp)
        self.embedded += '<span style="color:#000000; '
        self.embedded += f'background-color:#{bg};">'
        return

    def handle_endtag(self: EmbedExplanation, tag: str) -> None:
        assert(tag == 'span')
        self.embedded += '</span>'
        imp = self.importances[self.off]
        self.embedded += f'<span class="popup">{imp:.4f}</span>'
        self.embedded += '</a>'
        self.off += 1
        return

    def handle_data(self: EmbedExplanation, data: str) -> None:
        self.embedded += data
        return


class ProcessorGradCAM(Processor):
    def __init__(
        self: ProcessorGradCAM,
        config_json: str
    ) -> None:
        # call parent function
        self.config = ConfigProcessorGradCAM(config_json)
        return

    def _cache_preprocess(self: ProcessorGradCAM, preps: Dict) -> None:
        # call parent function
        super()._cache_preprocess(preps)
        prep = preps['prep']
        if self.config.mode == 'image':
            self.takeover_config = {
                'mode': self.config.mode,
            }
        else:  # self.config.mode == 'text'
            self.takeover_config = {
                'mode': self.config.mode,
                'data_name': prep.config.data_name,
                'base_dir': prep.config.base_dir,
                'num_class': prep.config.num_class,
                'word_len': prep.config.word_len,
                'word_vectors': prep.word_vectors,
            }
        self.unique_categories = prep.config.unique_categories
        return

    def _predict(
        self: ProcessorGradCAM,
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
        self: ProcessorGradCAM,
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

    def _predict_text(
        self: ProcessorGradCAM,
        dataloader: DataLoader,
        resources: List[Dict]
    ) -> None:
        category2label = {x: i for i, x in enumerate(self.unique_categories)}
        assert(len(self.config.target_categories) == 1)
        if self.config.target_categories[0] is None:
            target_label = None
        else:
            target_label = category2label[
                self.config.target_categories[0]
            ]
        pred_gcams = list()
        pred_cats = list()
        for batch in dataloader:
            batch_size = batch[0].size(0)
            if target_label is None:
                xpred = None
            else:
                xpred = np.array([target_label] * batch_size)
            res = self.model(batch[0].to(self.device), pred=xpred)
            for gcam, pred in zip(res['gcam'], res['pred']):
                pred_gcams.append(gcam)
                pred_cats.append(self.unique_categories[pred])
        for res, gcam, cat in zip(
            resources, pred_gcams, pred_cats
        ):
            embed = EmbedExplanation(gcam)
            embed.feed(res['spanned'])
            res['explain'] = embed.embedded
            res['explained_category'] = cat
            del res['spanned']
        return

    def _output_resources(
        self: Processor,
        resources: List[Dict],
        output_fname: str
    ) -> None:
        if self.config.mode == 'image':
            self._output_image(resources, output_fname)
        else:  # self.config.mode == 'text'
            self._output_json(resources, output_fname)
        return

    def _output_image(
        self: Processor,
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
