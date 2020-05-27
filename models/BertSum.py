#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Optional
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel
from .DeepLearning import ConfigDeepLearning

DEFAULT_RANDOM_STATE = None


class ConfigBertSum(ConfigDeepLearning):
    bertsum_params = [
        # name, vtype, is_require, default
        ('nict_bert_dir', str, True, None),
        ('random_state', int, False, DEFAULT_RANDOM_STATE),
    ]

    def __init__(
        self: ConfigBertSum,
        config: Dict,
        config_model_json: str
    ) -> None:
        # model_name
        config['model_name'] = 'BertSum'
        # load json
        self._load_one(config, config_model_json)
        # init parent class
        self._init_deeplearning(config)
        # set parameters
        for param in self.deeplearning_params:
            self._init_param(config, *param)
        for param in self.bertsum_params:
            self._init_param(config, *param)
        # value_assertion
        assert(os.path.exists(self.nict_bert_dir))
        return

    def load(self: ConfigBertSum) -> None:
        return

    def save(self: ConfigBertSum) -> None:
        return


class PositionalEncoder(nn.Module):
    def __init__(
        self: PositionalEncoder,
        config: BertConfig,
        max_length: Optional[int] = 5000
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        pos_enc = torch.zeros(max_length, config.hidden_size)
        pos = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(
                0, config.hidden_size, step=2, dtype=torch.float
            ) * -(np.log(10000.0) / config.hidden_size)
        )
        pos_enc[:, 0::2] = torch.sin(pos.float() * div_term)
        pos_enc[:, 1::2] = torch.cos(pos.float() * div_term)
        pos_enc = pos_enc.unsqueeze(0)
        self.register_buffer('pos_enc', pos_enc)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        return

    def forward(
        self: PositionalEncoder,
        embed: torch.Tensor,
        step: Optional[int] = None
    ) -> torch.Tensor:
        embed *= np.sqrt(self.hidden_size)
        if step is not None:
            embed += self.pos_enc[:, step][:, None, :]
        else:
            embed += self.pos_enc[:, :embed.size(1)]
        return self.dropout(embed)


class PositionwiseFeedForward(nn.Module):
    def __init__(
        self: PositionwiseFeedForward,
        config: BertConfig
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.linear1 = nn.Linear(
            config.hidden_size, config.intermediate_size
        )
        self.activate_func = F.gelu
        self.dropout1 = nn.Dropout(p=config.hidden_dropout_prob)
        self.linear2 = nn.Linear(
            config.intermediate_size, config.hidden_size
        )
        self.dropout2 = nn.Dropout(p=config.hidden_dropout_prob)
        return

    def forward(
        self: PositionwiseFeedForward,
        x: torch.Tensor
    ) -> torch.Tensor:
        inter = self.dropout1(
            self.activate_func(self.linear1(self.norm(x)))
        )
        output = self.dropout2(self.linear2(inter))
        return output + x


class BertSumAttention(nn.Module):
    def __init__(
        self: BertSumAttention,
        config: BertConfig
    ) -> None:
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads
        )
        all_head_size = self.num_attention_heads * self.attention_head_size
        self.all_head_size = all_head_size
        self.linear_query = nn.Linear(
            config.hidden_size, self.all_head_size
        )
        self.linear_keys = nn.Linear(
            config.hidden_size, self.all_head_size
        )
        self.linear_values = nn.Linear(
            config.hidden_size, self.all_head_size
        )
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.final_linear = nn.Linear(
            config.hidden_size, config.hidden_size
        )
        return

    def forward(
        self: BertSumAttention,
        data: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size = data.size(0)

        def shape(x):
            return x.view(
                batch_size,
                -1,
                self.num_attention_heads,
                self.attention_head_size
            ).transpose(1, 2)

        def unshape(x):
            return x.transpose(1, 2).contiguous().view(
                batch_size,
                -1,
                self.all_head_size
            )

        key = self.linear_keys(data)
        key = shape(key)
        value = self.linear_values(data)
        value = shape(value)
        query = self.linear_query(data)
        query = shape(query)
        query = query / np.sqrt(self.attention_head_size)
        scores = torch.matmul(query, key.transpose(2, 3))
        mask = mask.unsqueeze(1).expand_as(scores)
        scores = scores.masked_fill(mask, -1e18)
        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)
        context = unshape(torch.matmul(drop_attn, value))
        return self.final_linear(context)


class BertSumEncodeLayer(nn.Module):
    def __init__(
        self: BertSumEncodeLayer,
        config: BertConfig
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.attention = BertSumAttention(config)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.feed_forward = PositionwiseFeedForward(config)
        return

    def forward(
        self: BertSumEncodeLayer,
        layer_index: int,
        inputs: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        if layer_index == 0:
            input_norm = inputs
        else:
            input_norm = self.norm(inputs)
        mask = mask.unsqueeze(1)
        context = self.attention(input_norm, mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class BertSumEncoder(nn.Module):
    def __init__(
        self: BertSumEncoder,
        config: BertConfig
    ) -> None:
        super().__init__()
        self.config = config
        self.pos_encer = PositionalEncoder(config)
        self.encode_layers = nn.ModuleList([
            BertSumEncodeLayer(config) for _ in range(
                config.num_hidden_layers
            )
        ])
        self.norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.linear = nn.Linear(config.hidden_size, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        return

    def forward(
        self: BertSumEncoder,
        sentence_vector: torch.Tensor,
        cls_ids_mask: torch.Tensor,
        cls_ids_mask_bool: torch.Tensor
    ):
        sentence_length = sentence_vector.size(1)
        pos_emb = self.pos_encer.pos_enc[:, :sentence_length]
        x = sentence_vector + pos_emb
        for i, layer in enumerate(self.encode_layers):
            x = layer(i, x, ~cls_ids_mask_bool)
        scores = self.sigmoid(self.linear(self.norm(x)))
        return scores.squeeze(-1) * cls_ids_mask


class BertSum(nn.Module):
    def __init__(
        self: BertSum,
        config: Dict,
        config_model_json: str
    ) -> None:
        super().__init__()
        self.config = ConfigBertSum(config, config_model_json)
        # set random seed
        if self.config.random_state is not None:
            np.random.seed(self.config.random_state)
            torch.manual_seed(self.config.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.config.random_state)
        # load BertForSequenceClassification of NICT BERT
        bert_config = BertConfig.from_pretrained(
            os.path.join(self.config.nict_bert_dir, 'config.json')
        )
        if torch.cuda.is_available():
            map_location = torch.device('cuda')
        else:
            map_location = torch.device('cpu')
        state_dict = torch.load(
            os.path.join(self.config.nict_bert_dir, 'pytorch_model.bin'),
            map_location=map_location
        )
        self.bert = BertModel.from_pretrained(
            None,
            config=bert_config,
            state_dict=state_dict,
            from_tf=False
        )
        self.encoder = BertSumEncoder(bert_config)
        # disable training (changing its weights) of all bert layers
        for name, param in self.bert.named_parameters():
            param.requires_grad_(False)
        return

    def forward(
        self: BertSum,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        segment_ids: torch.Tensor,
        cls_ids: torch.Tensor,
        cls_ids_mask: torch.Tensor,
        cls_ids_mask_bool: torch.Tensor
    ) -> torch.Tensor:
        top_vector, _ = self.bert(input_ids, attention_mask, segment_ids)
        sentence_vector = top_vector[
            torch.arange(top_vector.size(0)).unsqueeze(1), cls_ids
        ]
        sentence_vector *= cls_ids_mask[:, :, None]
        scores = self.encoder(
            sentence_vector, cls_ids_mask, cls_ids_mask_bool
        ).squeeze(-1)
        return scores

    def load(self: BertSum) -> None:
        if torch.cuda.is_available():
            map_location = torch.device('cuda')
        else:
            map_location = torch.device('cpu')
        state_dict = torch.load(
            self.config.binary_path, map_location=map_location
        )
        self.encoder.load_state_dict(state_dict)
        return

    def save(self: BertSum) -> None:
        torch.save(
            self.encoder.state_dict(),
            self.config.binary_path
        )
        return

    def train(self: BertSum, mode: Optional[bool] = True) -> None:
        self.bert.train(mode)
        self.encoder.train(mode)
        return

    def eval(self: BertSum) -> None:
        self.bert.eval()
        self.encoder.eval()
        return
