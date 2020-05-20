#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Callable, Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import model_urls, Bottleneck, ResNet
from torch.hub import load_state_dict_from_url
from .DeepLearning import ConfigDeepLearning
from .TextCNN import TextCNN

GRADCAM_MODES = ['text', 'image']


class ConfigGradCAM(ConfigDeepLearning):
    gradcam_params = [
        # name, vtype, is_require, default
        ('mode', str, True, None),
        ('target_model_name', str, True, None),
        ('target_config_model_json', str, True, None),
        ('target_layer', str, False, None),
    ]

    def __init__(
        self: ConfigGradCAM,
        config: Dict,
        config_model_json: str
    ) -> None:
        # model_name
        config['model_name'] = 'GradCAM'
        # load json
        self._load_one(config, config_model_json)
        # set parameters
        for param in self.gradcam_params:
            self._init_param(config, *param)
        # value assertion
        assert(self.mode in GRADCAM_MODES)
        return

    def load(self: ConfigGradCAM) -> None:
        return

    def save(self: ConfigGradCAM) -> None:
        return


class GradCAM(nn.Module):
    def __init__(
        self: GradCAM,
        config: Dict,
        config_model_json: str
    ) -> None:
        super().__init__()
        self.config = ConfigGradCAM(config, config_model_json)
        # initialize
        self.target_layer_names = list()
        self.fmap_cache = dict()
        self.grad_cache = dict()
        self.handlers = list()
        # load target model of GradCAM
        if torch.cuda.is_available():
            map_location = torch.device('cuda')
        else:
            map_location = torch.device('cpu')
        if self.config.mode == 'image':
            if self.config.target_model_name == 'resnet152':
                self.model = ResNet(Bottleneck, [3, 8, 36, 3])
                state_dict = load_state_dict_from_url(
                    model_urls['resnet152'],
                    map_location=map_location,
                    progress=True
                )
                self.model.load_state_dict(state_dict)
            else:
                # not implemented
                assert(False)
        else:  # self.mode == 'text'
            if self.config.target_model_name == 'TextCNN':
                self.model = TextCNN(
                    config,
                    self.config.target_config_model_json
                )
            else:
                # not implemented
                assert(False)
            self.model.load()
        self.model.eval()
        self.model.to(map_location)
        return

    def _add_hook(self: GradCAM):
        # store forward activation mapping into fmap_cache
        def _save_fmap(key: str) -> Callable:
            def forward_hook(
                module: nn.Module,
                fmap_in: torch.Tensor,
                fmap_out: torch.Tensor
            ) -> None:
                self.fmap_cache[key] = fmap_out.detach()
                return
            return forward_hook

        # store gradient into grad_cache
        def _save_grad(key: str) -> Callable:
            def backward_hook(
                module: nn.Module,
                grad_in: torch.Tensor,
                grad_out: torch.Tensor
            ) -> None:
                self.grad_cache[key] = grad_out[0].detach()
                return
            return backward_hook

        # register above hook functions to the target layers
        for name, module in self.model.named_modules():
            if name == '':
                continue
            if not module._get_name().startswith("Conv"):
                continue
            if (
                self.config.target_layer is None
            ) or (
                self.config.target_layer in name
            ):
                self.target_layer_names.append(name)
                self.handlers.append(
                    module.register_forward_hook(_save_fmap(name))
                )
                self.handlers.append(
                    module.register_backward_hook(_save_grad(name))
                )
        # check whether the target layers are actually exist
        assert(len(self.target_layer_names) > 0)
        return

    def _remove_hook(self: GradCAM):
        for handler in self.handlers:
            handler.remove()
        self.handlers = list()
        return

    def _clear_cache(self: GradCAM) -> None:
        """clear cache to prepare next batch
        """
        self.fmap_cache = dict()
        self.grad_cache = dict()
        return

    def _get_gcam(
        self: GradCAM,
        target_layer: str,
        inter_shape: Tuple[int]
    ) -> torch.Tensor:
        """calc GradCAM of each target layer.
        """
        # weight of each output channel in target layer
        # weight of each output channel is the average of its gradient.
        weight = F.adaptive_avg_pool2d(self.grad_cache[target_layer], 1)
        fmap = self.fmap_cache[target_layer]
        # sum all weighted fmap of output channel by target layer
        # and ignore inactive (negative) target layer (ReLU)
        gcam = F.relu(torch.mul(fmap, weight).sum(dim=1, keepdim=True))
        # size of feature activation mapping(fmap) is less than
        # size(words/pixels/...) of input(sentense/picture/...)
        # by its kernel size.
        # So, interpolate fmap to make length the same as input
        return F.interpolate(
            gcam, inter_shape, mode='bilinear', align_corners=False
        )

    def __call__(
        self,
        x: torch.Tensor,
        pred: Optional[np.array]
    ) -> Dict:
        # add hook to target layer to retrieves fmap and gradients
        self._add_hook()
        # obtain the dimensions of input
        orig_shape = list(x.shape)
        # decide dimensions according to the mode
        if self.config.mode == "text":
            assert(len(orig_shape) == 2)
            inter_shape = tuple([orig_shape[1], 1])
            final_shape = tuple(orig_shape)
        else:  # mode == "image"
            assert(len(orig_shape) == 4)
            inter_shape = tuple(orig_shape[2:])
            final_shape = tuple([orig_shape[0]] + orig_shape[2:])
        # forward propagation (store forward activation mappings)
        logits = self.model(x.requires_grad_(False))
        probs = F.softmax(logits, dim=1)
        sorted_probs, ids = probs.sort(dim=1, descending=True)
        # create predicted class if that is not given
        # and retrive probability of given class (or maximun probability)
        if pred is None:
            prob = sorted_probs[:, 0].cpu().detach().numpy()
            pred = ids[:, 0]
        else:
            # check number of input sentences/pictures/... are the same
            assert(pred.shape[0] == logits.size(0))
            prob = list()
            for i in range(logits.size(0)):
                assert(pred[i] < logits.size(1))
                prob.append(probs[i, pred[i]].cpu().detach().numpy())
            pred = torch.unsqueeze(torch.from_numpy(pred), 1)
        # create one hot vector of predicted (or given) class
        one_hot = np.zeros((logits.size()), dtype=np.float32)
        for i in range(logits.size(0)):
            one_hot[i][pred[i]] = 1.0
        one_hot = torch.from_numpy(one_hot).requires_grad_(False)
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        one_hot = one_hot.to(device)
        # backward propagation (store gradients)
        self.model.zero_grad()
        logits.backward(gradient=one_hot, retain_graph=True)
        # get GradCAM of each target layer
        gcams = list()
        for target_layer in self.target_layer_names:
            gcams.append(
                self._get_gcam(target_layer, inter_shape)
            )
        # sum all GradCAM of target layers
        gcam = torch.cat(gcams, dim=1).sum(dim=1)
        # normalize
        gcam = gcam.view(orig_shape[0], -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(final_shape)
        # cleaning
        self._remove_hook()
        self._clear_cache()
        return {
            "pred": torch.squeeze(pred).cpu().detach().numpy(),
            "prob": prob,
            "gcam": gcam.cpu().detach().numpy(),
        }

    def load(self: GradCAM) -> None:
        return

    def save(self: GradCAM) -> None:
        return
