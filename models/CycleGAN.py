#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .DeepLearning import ConfigDeepLearning
from .NetworkGAN import GeneratorCycleGAN, DiscriminatorCycleGAN

DEFAULT_LAMBDA_A = 10.0
DEFAULT_LAMBDA_B = 10.0
DEFAULT_LAMBDA_IDENTITY = 0.5
DEFAULT_RANDOM_STATE = None


class ConfigCycleGAN(ConfigDeepLearning):
    cyclegan_params = [
        # name, vtype, is_require, default
        ('input_channels', int, True, None),
        ('output_channels', int, True, None),
        ('lambda_a', float, False, DEFAULT_LAMBDA_A),
        ('lambda_b', float, False, DEFAULT_LAMBDA_B),
        ('lambda_identity', float, False, DEFAULT_LAMBDA_IDENTITY),
        ('random_state', int, False, DEFAULT_RANDOM_STATE),
    ]

    def __init__(
        self: ConfigCycleGAN,
        config: Dict,
        config_model_json: str
    ) -> None:
        # model_name
        config['model_name'] = 'CycleGAN'
        # load json
        self._load_one(config, config_model_json)
        # init parent class
        self._init_deeplearning(config)
        # set parameters
        for param in self.deeplearning_params:
            self._init_param(config, *param)
        for param in self.cyclegan_params:
            self._init_param(config, *param)
        return

    def load(self: ConfigCycleGAN) -> None:
        # load json
        config = dict()
        self._load_one(config, self.config_json)
        # set parameters
        for param in self.deeplearning_params:
            self._init_param(config, *param)
        for param in self.cyclegan_params:
            self._init_param(config, *param)
        return

    def save(self: ConfigCycleGAN) -> None:
        config = dict()
        # save parameters
        for name, _, _, _ in self.deeplearning_params:
            self._save_param(config, name)
        for name, _, _, _ in self.cyclegan_params:
            self._save_param(config, name)
        self._save(config)
        return


def _init_weights(net: nn.Module, std: Optional[float] = 0.02) -> None:
    def _init_func(m: nn.Module):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and ((
            classname.find('Conv') != -1
        ) or (
            classname.find('Linear') != -1
        )):
            init.normal_(m.weight.data, mean=0.0, std=std)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, mean=1.0, std=std)
            init.constant_(m.bias.data, 0.0)
        return
    net.apply(_init_func)
    return


class DataPool(object):
    def __init__(self: DataPool, pool_size: Optional[int] = 64) -> None:
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.data_list = list()
        return

    def query(self: DataPool, data: torch.Tensor) -> torch.Tensor:
        if self.pool_size == 0:
            return data
        return_data = list()
        for element in data.data:
            element = torch.unsqueeze(element, dim=0)
            if len(self.data_list) < self.pool_size:
                self.data_list.append(element)
                return_data.append(element)
            else:
                p = np.random.uniform(0.0, 1.0, 1)[0]
                if p > 0.5:
                    idx = np.random.randint(0, self.pool_size, 1)[0]
                    tmp = self.data_list[idx].clone()
                    self.data_list[idx] = element
                    return_data.append(tmp)
                else:
                    return_data.append(element)
        return torch.cat(return_data, dim=0)


class CycleGAN(nn.Module):
    def __init__(
        self: CycleGAN,
        config: Dict,
        config_model_json: str
    ) -> None:
        super().__init__()
        self.config = ConfigCycleGAN(config, config_model_json)
        # set random seed
        if self.config.random_state is not None:
            np.random.seed(self.config.random_state)
            torch.manual_seed(self.config.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.config.random_state)
        # generators
        self.genAB = GeneratorCycleGAN(
            in_channels=self.config.input_channels,
            out_channels=self.config.output_channels
        )
        self.genBA = GeneratorCycleGAN(
            in_channels=self.config.input_channels,
            out_channels=self.config.output_channels
        )
        # discriminators
        self.disA = DiscriminatorCycleGAN(
            in_channels=self.config.output_channels
        )
        self.disB = DiscriminatorCycleGAN(
            in_channels=self.config.output_channels
        )
        # initialize weights
        _init_weights(self.genAB)
        _init_weights(self.genBA)
        _init_weights(self.disA)
        _init_weights(self.disB)
        # data pool
        self.poolA = DataPool()
        self.poolB = DataPool()
        return

    def forward(
        self: CycleGAN,
        realA: torch.Tensor,
        realB: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        # Generator's loss 1: Identity loss
        lossAA = F.l1_loss(
            self.genBA(realA), realA
        ).cpu() * self.config.lambda_a * self.config.lambda_identity
        lossBB = F.l1_loss(
            self.genAB(realB), realB
        ).cpu() * self.config.lambda_a * self.config.lambda_identity
        loss_identity = (lossAA + lossBB) * 0.5
        # Generator's loss 2: GAN loss
        fakeB = self.genAB(realA)
        pred_AB = self.disB(fakeB).cpu()
        lossAB = F.mse_loss(
            pred_AB,
            torch.tensor(1.0).expand_as(pred_AB)
        )
        fakeA = self.genBA(realB)
        pred_BA = self.disA(fakeA).cpu()
        lossBA = F.mse_loss(
            pred_BA,
            torch.tensor(1.0).expand_as(pred_BA)
        )
        loss_gan = (lossAB + lossBA) * 0.5
        # Generator's loss 3: Cycle loss
        cycleA = self.genBA(fakeB)
        lossABA = F.l1_loss(
            cycleA, realA
        ).cpu() * self.config.lambda_a
        cycleB = self.genAB(fakeA)
        lossBAB = F.l1_loss(
            cycleB, realB
        ).cpu() * self.config.lambda_b
        loss_cycle = (lossABA + lossBAB) * 0.5
        # Generator's loss
        loss_gen = loss_gan + loss_cycle + loss_identity
        # keep fakes for forward_discriminator()
        self.fakeA = fakeA.detach()
        self.fakeB = fakeB.detach()
        return loss_gen

    def forward_discriminator(
        self: CycleGAN,
        side: str,
        real: torch.Tensor
    ) -> torch.Tensor:
        if side == 'A':
            dis = self.disA
            pool = self.poolA
            fake = self.fakeA
        else:  # side == 'B'
            dis = self.disB
            pool = self.poolB
            fake = self.fakeB
        # Discriminator's loss 1: real loss
        pred_real = dis(real).cpu()
        loss_real = F.mse_loss(
            pred_real,
            torch.tensor(1.0).expand_as(pred_real)
        )
        # Discriminator's loss 2: fake loss
        fake_ = pool.query(fake)
        pred_fake = dis(fake_).cpu()
        loss_fake = F.mse_loss(
            pred_fake,
            torch.tensor(0.0).expand_as(pred_fake)
        )
        # Discriminator's loss
        loss_dis = (loss_real + loss_fake) * 0.5
        return loss_dis

    def __call__(
        self: CycleGAN,
        dataA: Optional[torch.Tensor],
        dataB: Optional[torch.Tensor]
    ) -> Dict:
        ret = dict()
        if dataA is not None:
            ret['fakeB'] = self.genAB(dataA)
        if dataB is not None:
            ret['fakeA'] = self.genBA(dataB)
        return ret

    def load(self: CycleGAN) -> None:
        self.config.load()
        if torch.cuda.is_available():
            map_location = torch.device('cuda')
        else:
            map_location = torch.device('cpu')
        self.load_state_dict(torch.load(
            self.config.binary_path, map_location=map_location
        ))
        return

    def save(self: CycleGAN) -> None:
        self.config.save()
        torch.save(self.state_dict(), self.config.binary_path)
        return

    def train(self: CycleGAN, mode: bool) -> None:
        self.genAB.train(mode)
        self.genBA.train(mode)
        self.disA.train(mode)
        self.disB.train(mode)
        return

    def eval(self: CycleGAN) -> None:
        self.genAB.eval()
        self.genBA.eval()
        self.disA.eval()
        self.disB.eval()
        return
