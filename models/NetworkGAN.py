#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional
from collections import OrderedDict
import torch
import torch.nn as nn


class GeneratorCycleGAN(nn.Module):
    def __init__(
        self: GeneratorCycleGAN,
        in_channels: int,
        out_channels: int,
        num_filters: Optional[int] = 64,
        num_layers: Optional[int] = 9,
        num_sampling: Optional[int] = 2
    ) -> None:
        super().__init__()
        modules = OrderedDict()
        modules['pre_pad'] = nn.ReflectionPad2d(in_channels)
        modules['pre_conv'] = nn.Conv2d(
            in_channels=in_channels, out_channels=num_filters,
            kernel_size=7,
            padding=0, bias=False
        )
        modules['pre_norm'] = nn.BatchNorm2d(
            num_features=num_filters,
            affine=True, track_running_stats=True
        )
        modules['pre_relu'] = nn.ReLU(inplace=True)
        for i in range(num_sampling):
            mult = 2 ** i
            modules['down_%d' % i] = DownSamplingLayer(
                in_channels=num_filters * mult,
                out_channels=num_filters * mult * 2
            )
        dim = num_filters * (2 ** num_sampling)
        for i in range(num_layers):
            modules['resnet_%d' % i] = ResnetLayer(dim=dim)
        for i in range(num_sampling):
            mult = 2 ** (num_sampling - i)
            modules['up_%d' % i] = UpSamplingLayer(
                in_channels=num_filters * mult,
                out_channels=num_filters * mult // 2
            )
        modules['post_pad'] = nn.ReflectionPad2d(out_channels)
        modules['post_conv'] = nn.Conv2d(
            in_channels=num_filters, out_channels=out_channels,
            kernel_size=7,
            padding=0, bias=False
        )
        modules['tanh'] = nn.Tanh()
        self.net = nn.Sequential(modules)
        return

    def forward(self: GeneratorCycleGAN, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownSamplingLayer(nn.Module):
    def __init__(
        self: DownSamplingLayer,
        in_channels: int,
        out_channels: int
    ) -> None:
        super().__init__()
        modules = OrderedDict()
        modules['conv'] = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=3,
            padding=1, bias=False
        )
        modules['norm'] = nn.BatchNorm2d(
            num_features=out_channels,
            affine=True, track_running_stats=True
        )
        modules['relu'] = nn.ReLU(inplace=True)
        self.net = nn.Sequential(modules)
        return

    def forward(self: DownSamplingLayer, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpSamplingLayer(nn.Module):
    def __init__(
        self: UpSamplingLayer,
        in_channels: int,
        out_channels: int
    ) -> None:
        super().__init__()
        modules = OrderedDict()
        modules['conv1'] = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=3,
            padding=1, output_padding=1, bias=False
        )
        modules['norm'] = nn.BatchNorm2d(
            num_features=out_channels,
            affine=True, track_running_stats=True
        )
        modules['relu'] = nn.ReLU(inplace=True)
        self.net = nn.Sequential(modules)
        return

    def forward(self: UpSamplingLayer, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResnetLayer(nn.Module):
    def __init__(self: ResnetLayer, dim: int) -> None:
        """Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super().__init__()
        modules = OrderedDict()
        modules['pad1'] = nn.ReflectionPad2d(1)
        modules['conv1'] = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=3,
            padding=0, bias=False
        )
        modules['norm1'] = nn.BatchNorm2d(
            num_features=dim,
            affine=True, track_running_stats=True
        )
        modules['relu'] = nn.ReLU(inplace=True)
        modules['pad2'] = nn.ReflectionPad2d(1)
        modules['conv2'] = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=3,
            padding=0, bias=False
        )
        modules['norm2'] = nn.BatchNorm2d(
            num_features=dim,
            affine=True, track_running_stats=True
        )
        self.net = nn.Sequential(modules)
        return

    def forward(self: ResnetLayer, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class DiscriminatorCycleGAN(nn.Module):
    def __init__(
        self: DiscriminatorCycleGAN,
        in_channels: int,
        num_filters: Optional[int] = 64,
        num_layers: Optional[int] = 4
    ) -> None:
        super().__init__()
        modules = OrderedDict()
        layer_in = in_channels
        layer_out = num_filters
        for i in range(num_layers):
            modules['%d_conv' % i] = nn.Conv2d(
                in_channels=layer_in, out_channels=layer_out,
                kernel_size=4,
                stride=2, padding=1
            )
            if i > 0:
                modules['%d_norm' % i] = nn.BatchNorm2d(
                    num_features=layer_out,
                    affine=True, track_running_stats=True
                )
            modules['%d_relu' % i] = nn.LeakyReLU(
                negative_slope=0.2, inplace=True
            )
            layer_in = layer_out
            if layer_out < 512:
                layer_out *= 2
        modules['conv'] = nn.Conv2d(
            in_channels=layer_in, out_channels=1,
            kernel_size=4,
            stride=1, padding=1
        )
        self.net = nn.Sequential(modules)
        return

    def forward(
        self: DiscriminatorCycleGAN,
        x: torch.Tensor
    ) -> torch.Tensor:
        return self.net(x)
