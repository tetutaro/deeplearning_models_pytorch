#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
load weight file of original implemented model to this model
1. copy this file to the root directory of this repository
    > cp load_cyclegan.py ../.
2. download pretrained weights using pytorch_CygleGAN_and_pix2pix
    > cd <somewhare>
    > git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
    > cd pytorch_CycleGAN_and_pix2pix
    > sh scripts/download_cyclegan_model.sh <modelA>
    > sh scripts/download_cyclegan_model.sh <modelB>
3. invoke this script
    > ./load_cyclegan.py -c configs/... -a <path to modelA> -b <path to modelB>
"""
from __future__ import annotations
import os
import click
import simplejson as json
import torch
from models.CycleGAN import CycleGAN


def patch_norm(state_dict, module, keys, i=0):
    key = keys[i]
    if i + 1 == len(keys):
        if module.__class__.__name__.startswith('InstanceNorm') and \
                key == 'running_mean' or key == 'running_var':
            if getattr(module, key) is None:
                state_dict.pop('.'.join(keys))
        if module.__class__.__name__.startswith('InstanceNorm') and \
                key == 'num_batches_tracked':
            state_dict.pop('.'.join(keys))
    else:
        patch_norm(state_dict, getattr(module, key), keys, i+1)
    return


def rearrange_state_dict(state_dict, net):
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
    for key in list(state_dict.keys()):
        patch_norm(state_dict, net, key.split('.'))
    return


@click.command()
@click.option('--config', '-c', required=True, type=str)
@click.option('--modela', '-a', required=True, type=str)
@click.option('--modelb', '-b', required=True, type=str)
def main(config: str, modela: str, modelb: str) -> None:
    # assertion
    assert(os.path.exists(modela))
    assert(os.path.exists(modelb))
    # prepare config
    with open(config, 'rt') as rf:
        config_total = json.load(rf)
    config_data_json = config_total['config_data_json']
    with open(config_data_json, 'rt') as rf:
        config_data = json.load(rf)
    config_model_json = config_total['config_model_json']
    config_takeover = dict()
    image_dir = config_data['image_dir']
    if image_dir.endswith(os.sep):
        image_dir = image_dir[:-1]
    data_name = image_dir.split(os.sep)[-1]
    base_dir = os.path.join(
        'binaries', 'CycleGAN', data_name
    )
    os.makedirs(base_dir, exist_ok=True)
    config_takeover['data_name'] = data_name
    config_takeover['base_dir'] = base_dir
    # load model
    model = CycleGAN(config_takeover, config_model_json)
    state_dict = torch.load(modela, map_location='cpu')
    rearrange_state_dict(state_dict, model.genAB)
    model.genAB.load_state_dict(state_dict)
    state_dict = torch.load(modelb, map_location='cpu')
    rearrange_state_dict(state_dict, model.genBA)
    model.genBA.load_state_dict(state_dict)
    # save weights
    model.eval()
    model.save()
    return


if __name__ == "__main__":
    main()
