#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Optional
import os
import click
import simplejson as json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
from models.ImageLoaderCycleGAN import ImageLoaderCycleGAN
from models.CycleGAN import CycleGAN


class DataCycleGAN(object):
    def __init__(
        self: DataCycleGAN,
        pathA: str,
        pathB: str,
        device: torch.Device
    ) -> None:
        self._load_data(pathA, 'A', device)
        self._load_data(pathB, 'B', device)
        return

    def _load_data(
        self: DataCycleGAN,
        path: str,
        side: str,
        device: torch.Device
    ) -> None:
        real = 'real%s' % side
        data = 'data%s' % side
        compose = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])
        if path is None:
            setattr(self, real, None)
            setattr(self, data, None)
            return
        reald = Image.open(path).convert('RGB')
        setattr(self, real, reald)
        rdata = compose(reald.copy()).unsqueeze(dim=0).to(device)
        setattr(self, data, rdata)
        return

    def convert_fake(
        self: DataCycleGAN,
        ret: Dict,
        side: str,
        name: str
    ) -> Optional[np.array]:
        setattr(self, 'name%s' % side, name)
        oside = 'A' if side == 'B' else 'B'
        fname = 'fake%s' % oside
        compose = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256))
        ])
        fake = ret.get(fname)
        if fake is None:
            setattr(self, fname, None)
            return
        fdata = np.array(
            compose(fake.cpu().squeeze(dim=0).detach()),
            dtype=np.uint8
        )
        setattr(self, fname, fdata)
        return

    def output_each(
        self: DataCycleGAN,
        side: str
    ) -> None:
        oside = 'A' if side == 'B' else 'B'
        fake = getattr(self, 'fake%s' % oside)
        if fake is None:
            return
        fname = 'results/CycleGAN_%s_fake_%s.png' % (
            self.data_name,
            getattr(self, 'name%s' % oside)
        )
        fig = plt.figure(figsize=(2.56, 2.56))
        plt.subplots_adjust(
            top=1, bottom=0, right=1, left=0, hspace=0, wspace=0
        )
        ax = plt.gca()
        ax.imshow(fake)
        ax.set_axis_off()
        plt.margins(0, 0)
        plt.savefig(fname, pad_inches=0)
        plt.close(fig)
        return

    def output_grid(
        self: DataCycleGAN,
    ) -> None:
        imgs = list()
        titles = list()
        hnum = 0
        wnum = 2
        if self.realA is not None:
            imgs.append(self.realA)
            titles.append('real %s' % self.nameA)
            imgs.append(self.fakeB)
            titles.append('fake %s' % self.nameB)
            hnum += 1
        if self.realB is not None:
            imgs.append(self.realB)
            titles.append('real %s' % self.nameB)
            imgs.append(self.fakeA)
            titles.append('fake %s' % self.nameA)
            hnum += 1
        wnum = 2
        fig = plt.figure(figsize=(wnum * 2.56, hnum * 2.56), facecolor='w')
        for i, (img, title) in enumerate(zip(imgs, titles)):
            ax = fig.add_subplot(hnum, wnum, i + 1)
            ax.imshow(img)
            ax.set_axis_off()
            ax.set_title(title)
        plt.savefig(
            'results/CycleGAN_%s.png' % self.data_name,
            facecolor='w', edgecolor='w',
            bbox_inches='tight', pad_inches=0.1
        )
        return


class ModelCycleGAN(object):
    def __init__(
        self: ModelCycleGAN,
        config: str,
        device: torch.Device
    ) -> None:
        with open(config, 'rt') as rf:
            process_config = json.load(rf)
        prep = ImageLoaderCycleGAN(
            process_config['config_data_json'],
            process_config['config_preprocessor_json']
        )
        takeover_config = {
            'data_name': prep.config.data_name,
            'base_dir': prep.config.base_dir,
        }
        self.model = CycleGAN(
            takeover_config,
            process_config['config_model_json']
        )
        self.model.load()
        self.model.eval()
        self.model.to(device)
        self.nameA = prep.config.name_a
        self.nameB = prep.config.name_b
        self.data_name = prep.config.data_name
        return

    def generate(
        self: ModelCycleGAN,
        data: DataCycleGAN
    ) -> None:
        data.data_name = self.data_name
        with torch.no_grad():
            ret = self.model(dataA=data.dataA, dataB=data.dataB)
        data.convert_fake(ret, 'A', self.nameA)
        data.convert_fake(ret, 'B', self.nameB)
        return


@click.command()
@click.option('--config', '-c', required=True, type=str)
@click.option('--a-path', '-a', type=str)
@click.option('--b-path', '-b', type=str)
@click.option('--grid/--no-grid', is_flag=True, default=True)
def main(config: str, a_path: str, b_path: str, grid: bool) -> None:
    # assertion
    assert(os.path.exists(config))
    if a_path is not None:
        assert(os.path.exists(a_path))
    if b_path is not None:
        assert(os.path.exists(b_path))
    if a_path is None and b_path is None:
        raise(ValueError("specify at least either A or B"))
    # detect device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # load CycleGAN
    model = ModelCycleGAN(config, device)
    # prepare data
    data = DataCycleGAN(a_path, b_path, device)
    # generate fake
    model.generate(data)
    # output
    if grid:
        data.output_grid()
    else:
        data.output_each('A')
        data.output_each('B')
    return


if __name__ == "__main__":
    main()
