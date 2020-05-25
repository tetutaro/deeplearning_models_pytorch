#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Tuple
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


def load_image(path: str) -> Tuple[torch.Tensor, np.array, Tuple[int]]:
    raw = Image.open(path).convert('RGB')
    compose = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    img = compose(raw.copy()).unsqueeze(dim=0)
    size = raw.size
    return img, np.array(raw, dtype=np.uint8), size


def convert_fake(fake: torch.Tensor, size: Tuple[int]) -> np.array:
    img = fake.cpu().squeeze(dim=0).detach()
    compose = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size)
    ])
    img = compose(img)
    return np.array(img, dtype=np.uint8)


@click.command()
@click.option('--config', '-c', required=True, type=str)
@click.option('--a-path', '-a', type=str)
@click.option('--b-path', '-b', type=str)
def main(config: str, a_path: str, b_path: str) -> None:
    assert(os.path.exists(config))
    if a_path is not None:
        assert(os.path.exists(a_path))
    if b_path is not None:
        assert(os.path.exists(b_path))
    if a_path is None and b_path is None:
        raise(ValueError("specify at least either A or B"))
    with open(config, 'rt') as rf:
        process_config = json.load(rf)
    prep = ImageLoaderCycleGAN(
        process_config['config_data_json'],
        process_config['config_preprocessor_json']
    )
    output_prefix = os.path.join(
        'results',
        "_".join([
            prep.config.model_name,
            prep.config.data_name
        ])
    )
    output_extension = process_config['output_extension']
    output_fname = output_prefix + '_'
    if a_path is not None:
        output_fname += os.path.splitext(
            os.path.basename(a_path)
        )[0]
        if b_path is not None:
            output_fname += '_'
    if b_path is not None:
        output_fname += os.path.splitext(
            os.path.basename(b_path)
        )[0]
    output_fname += '.' + output_extension
    takeover_config = {
        'data_name': prep.config.data_name,
        'base_dir': prep.config.base_dir,
    }
    model = CycleGAN(takeover_config, process_config['config_model_json'])
    model.load()
    model.eval()
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.to(device)
    if a_path is None:
        dataA = None
    else:
        dataA, rawA, sizeA = load_image(a_path)
        dataA = dataA.to(device)
    if b_path is None:
        dataB = None
    else:
        dataB, rawB, sizeB = load_image(b_path)
        dataB = dataB.to(device)
    with torch.no_grad():
        ret = model(dataA=dataA, dataB=dataB)
    if dataA is not None:
        fakeB = convert_fake(ret['fakeB'], sizeA)
    if dataB is not None:
        fakeA = convert_fake(ret['fakeA'], sizeB)
    if dataA is None or dataB is None:
        hnum = 1
    else:
        hnum = 2
    wnum = 2
    fig = plt.figure(figsize=(wnum * 5, hnum * 5), facecolor='w')
    off = 1
    if dataA is not None:
        ax = fig.add_subplot(hnum, wnum, off)
        ax.imshow(rawA)
        ax.set_axis_off()
        ax.set_title('realA')
        off += 1
        ax = fig.add_subplot(hnum, wnum, off)
        ax.imshow(fakeB)
        ax.set_axis_off()
        ax.set_title('fakeB')
        off += 1
    if dataB is not None:
        ax = fig.add_subplot(hnum, wnum, off)
        ax.imshow(rawB)
        ax.set_axis_off()
        ax.set_title('realB')
        off += 1
        ax = fig.add_subplot(hnum, wnum, off)
        ax.imshow(fakeA)
        ax.set_axis_off()
        ax.set_title('fakeA')
        off += 1
    plt.savefig(
        output_fname,
        facecolor='w', edgecolor='w',
        bbox_inches='tight', pad_inches=0.1
    )
    return


if __name__ == "__main__":
    main()
