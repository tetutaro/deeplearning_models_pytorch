#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn as nn


class GeneratorCycleGAN(nn.Module):
    def __init__(
        self: GeneratorCycleGAN,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        num_layers: int,
        num_sampling: int
    ) -> None:
        super().__init__()
        modules = list()
        modules.append(nn.ReflectionPad2d(in_channels))
        modules.append(nn.Conv2d(
            in_channels=in_channels, out_channels=num_filters, kernel_size=7,
            stride=1, padding=0, bias=True
        ))
        modules.append(nn.InstanceNorm2d(
            num_features=num_filters,
            affine=False, track_running_stats=False
        ))
        modules.append(nn.ReLU(inplace=True))
        for i in range(num_sampling):
            mult = 2 ** i
            modules.append(nn.Conv2d(
                in_channels=num_filters * mult,
                out_channels=num_filters * mult * 2,
                kernel_size=3,
                stride=2, padding=1, bias=True
            ))
            modules.append(nn.InstanceNorm2d(
                num_features=num_filters * mult * 2,
                affine=False, track_running_stats=False
            ))
            modules.append(nn.ReLU(inplace=True))
        dim = num_filters * (2 ** num_sampling)
        for _ in range(num_layers):
            modules.append(ResnetBlock(dim=dim))
        for i in range(num_sampling):
            mult = 2 ** (num_sampling - i)
            modules.append(nn.ConvTranspose2d(
                in_channels=num_filters * mult,
                out_channels=num_filters * mult // 2,
                kernel_size=3,
                stride=2, padding=1, output_padding=1, bias=True
            ))
            modules.append(nn.InstanceNorm2d(
                num_features=num_filters * mult // 2,
                affine=False, track_running_stats=False
            ))
            modules.append(nn.ReLU(inplace=True))
        modules.append(nn.ReflectionPad2d(out_channels))
        modules.append(nn.Conv2d(
            in_channels=num_filters, out_channels=out_channels, kernel_size=7,
            stride=1, padding=0, bias=True
        ))
        modules.append(nn.Tanh())
        self.model = nn.Sequential(*modules)
        return

    def forward(self: GeneratorCycleGAN, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ResnetBlock(nn.Module):
    def __init__(self: ResnetBlock, dim: int) -> None:
        """Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super().__init__()
        modules = list()
        modules.append(nn.ReflectionPad2d(1))
        modules.append(nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=3,
            stride=1, padding=0, bias=True
        ))
        modules.append(nn.InstanceNorm2d(
            num_features=dim,
            affine=False, track_running_stats=False
        ))
        modules.append(nn.ReLU(inplace=True))
        modules.append(nn.ReflectionPad2d(1))
        modules.append(nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=3,
            stride=1, padding=0, bias=True
        ))
        modules.append(nn.InstanceNorm2d(
            num_features=dim,
            affine=False, track_running_stats=False
        ))
        self.conv_block = nn.Sequential(*modules)
        return

    def forward(self: ResnetBlock, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class DiscriminatorCycleGAN(nn.Module):
    def __init__(
        self: DiscriminatorCycleGAN,
        in_channels: int,
        num_filters: int,
        num_layers: int
    ) -> None:
        super().__init__()
        modules = list()
        modules.append(nn.Conv2d(
            in_channels=in_channels, out_channels=num_filters, kernel_size=4,
            stride=2, padding=1, bias=True
        ))
        modules.append(nn.LeakyReLU(
            negative_slope=0.2, inplace=True
        ))
        prev_mult = 1
        mult = 1
        for i in range(1, num_layers):
            prev_mult = mult
            mult = min(2 ** i, 8)
            modules.append(nn.Conv2d(
                in_channels=num_filters * prev_mult,
                out_channels=num_filters * mult,
                kernel_size=4,
                stride=2, padding=1, bias=True
            ))
            modules.append(nn.InstanceNorm2d(
                num_features=num_filters * mult,
                affine=False, track_running_stats=False
            ))
            modules.append(nn.LeakyReLU(
                negative_slope=0.2, inplace=True
            ))
        prev_mult = mult
        mult = min(2 ** num_layers, 8)
        modules.append(nn.Conv2d(
            in_channels=num_filters * prev_mult,
            out_channels=num_filters * mult,
            kernel_size=4,
            stride=1, padding=1, bias=True
        ))
        modules.append(nn.InstanceNorm2d(
            num_features=num_filters * mult,
            affine=False, track_running_stats=False
        ))
        modules.append(nn.LeakyReLU(
            negative_slope=0.2, inplace=True
        ))
        modules.append(nn.Conv2d(
            in_channels=num_filters * mult, out_channels=1, kernel_size=4,
            stride=1, padding=1, bias=True
        ))
        self.model = nn.Sequential(*modules)
        return

    def forward(
        self: DiscriminatorCycleGAN,
        x: torch.Tensor
    ) -> torch.Tensor:
        return self.model(x)
