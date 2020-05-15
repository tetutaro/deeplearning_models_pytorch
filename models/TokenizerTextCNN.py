#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple
import torch
from torch.utils.data import TensorDataset
from .Tokenizer import ConfigTokenizer, Tokenizer

DEFAULT_MIN_WORD_LEN = 3
DEFAULT_MAX_WORD_LEN = 512


class ConfigTokenizerTextCNN(ConfigTokenizer):
    tokenizer_textcnn_params = [
        # name, vtype, is_require, default
        ('model_name', str, True, None),
        ('document_column', str, True, None),
        ('category_column', str, True, None),
        ('min_word_len', int, False, DEFAULT_MIN_WORD_LEN),
        ('max_word_len', int, False, DEFAULT_MAX_WORD_LEN),
        ('word_len', int, False, 0),
    ]

    def __init__(
        self: ConfigTokenizerTextCNN,
        config_data_json: str,
        config_preprocess_json: str
    ) -> None:
        # model_name
        config = dict()
        config['model_name'] = 'TextCNN'
        # call parent function
        super().__init__(
            config,
            config_data_json,
            config_preprocess_json
        )
        # set parameters
        for param in self.tokenizer_textcnn_params:
            self.init_param(config, *param)
        # value assertion
        assert((
            self.min_word_len > 0
        ) and (
            self.min_word_len <= self.max_word_len
        ))
        assert((
            self.max_word_len > 0
        ) and (
            self.max_word_len <= DEFAULT_MAX_WORD_LEN
        ))
        return

    def load(
        self: ConfigTokenizerTextCNN,
        config_json: str
    ) -> None:
        # call parent function
        config = super().load(config_json)
        # set parameters
        for param in self.tokenizer_textcnn_params:
            self.init_param(config, *param)
        return

    def save(self: ConfigTokenizerTextCNN) -> None:
        config = dict()
        # save parameters
        for name, _, _, _ in self.tokenizer_textcnn_params:
            self.save_param(config, name)
        # call parent function
        return super().save(config)

    def get_max_word_len(
        self: ConfigTokenizerTextCNN,
        is_encoded: bool
    ) -> int:
        if is_encoded:
            max_word_len = self.word_len
        else:
            max_word_len = self.max_word_len
        return max_word_len

    def save_word_len(
        self: ConfigTokenizerTextCNN,
        word_len: int
    ) -> None:
        self.word_len = word_len
        return


class TokenizerTextCNN(Tokenizer):
    def __init__(
        self: TokenizerTextCNN,
        config: ConfigTokenizerTextCNN
    ) -> None:
        super().__init__(config)
        self.config = config
        self.load_mecab()
        self.load_keyedvectors()
        return

    def save(self: TokenizerTextCNN) -> None:
        self.config.save()
        return

    def preprocess(
        self: TokenizerTextCNN
    ) -> Tuple[TensorDataset, List[Dict]]:
        is_encoded = self.config.is_encoded()
        max_word_len = self.config.get_max_word_len(is_encoded)
        # read input_json_path and extract documents and labels
        original_list = list()
        spanned_list = list()
        word_ids_list = list()
        category_list = list()
        for data in self.yield_data_json():
            document = data.get(self.config.document_column)
            category = data.get(self.config.category_column)
            if document is None:
                continue
            if category is None:
                if is_encoded:
                    category = ''
                else:
                    continue
            preprocessed = self._preprocess_each_document(
                document, max_word_len
            )
            if len(preprocessed) == 0:
                continue
            if len(preprocessed['word_ids']) < self.config.min_word_len:
                continue
            original_list.append(preprocessed['original'])
            spanned_list.append(preprocessed['spanned'])
            word_ids_list.append(preprocessed['word_ids'])
            category_list.append(category)
        # produce labels by encoding categories
        label_list = self.encode_categories(category_list)
        word_len, word_ids_list = self.padding_list(
            word_ids_list, None
        )
        if not is_encoded:
            self.config.save_word_len(word_len)
        word_ids_list = torch.tensor(word_ids_list).long()
        label_list = torch.tensor(label_list)
        dataset = TensorDataset(
            word_ids_list, label_list
        )
        # summarize original data
        documents = list()
        for o, s, c in zip(
            original_list, spanned_list, category_list
        ):
            documents.append({
                'document': o, 'spanned': s, 'category': c
            })
        return dataset, documents

    def _preprocess_each_document(
        self: TokenizerTextCNN,
        document: str,
        max_word_len: int
    ) -> Dict:
        original = ''
        spanned = ''
        word_ids = list()
        # split document into nodes using MeCab
        for node in self.yield_parsed_node(
            self.cleaning_text(document)
        ):
            surf, word_id = self.get_word_id(node)
            if surf is None:
                continue
            original += surf
            if word_id is None:
                spanned += surf
                continue
            if len(word_ids) == max_word_len:
                spanned += surf
                continue
            spanned += f"<span>{surf}</span>"
            word_ids.append(word_id)
        return {
            'original': original,
            'spanned': spanned,
            'word_ids': word_ids,
        }
