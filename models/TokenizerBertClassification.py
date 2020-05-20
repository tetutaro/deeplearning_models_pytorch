#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import os
import torch
from torch.utils.data import TensorDataset
from .Tokenizer import ConfigTokenizer, Tokenizer

DEFAULT_MIN_WORD_LEN = 3
DEFAULT_MAX_WORD_LEN = 510
DEFAULT_MIN_SENTENCES = 3
DEFAULT_MAX_SENTENCES = 100


class ConfigTokenizerBertClassification(ConfigTokenizer):
    tokenizer_bert_classification_params = [
        # name, vtype, is_require, default
        ('model_name', str, True, None),
        ('document_column', str, True, None),
        ('category_column', str, True, None),
        ('nict_bert_dir', str, True, None),
        ('min_word_len', int, False, DEFAULT_MIN_WORD_LEN),
        ('max_word_len', int, False, DEFAULT_MAX_WORD_LEN),
        ('min_sentences', int, False, DEFAULT_MIN_SENTENCES),
        ('max_sentences', int, False, DEFAULT_MAX_SENTENCES),
    ]

    def __init__(
        self: ConfigTokenizerBertClassification,
        config_data_json: str,
        config_preprocess_json: str
    ) -> None:
        # model_name
        config = dict()
        config['model_name'] = 'BertClassification'
        self._load_two(config, config_data_json, config_preprocess_json)
        # initialize parent class
        self._init_preprocessor(config)
        self._init_tokenizer(config)
        # set parameters
        for param in self.preprocessor_params:
            self._init_param(config, *param)
        for param in self.tokenizer_params:
            self._init_param(config, *param)
        for param in self.tokenizer_bert_classification_params:
            self._init_param(config, *param)
        # value assertion
        assert((
            self.min_word_len > 0
        ) and (
            self.min_word_len <= self.max_word_len
        ))
        assert(self.max_word_len <= DEFAULT_MAX_WORD_LEN)
        assert((
            self.min_sentences > 0
        ) and (
            self.min_sentences <= self.max_sentences
        ))
        assert(os.path.exists(self.nict_bert_dir))
        return

    def load(self: ConfigTokenizerBertClassification) -> None:
        return

    def save(self: ConfigTokenizerBertClassification) -> None:
        return


class TokenizerBertClassification(Tokenizer):
    def __init__(
        self: TokenizerBertClassification,
        config_data_json: str,
        config_preprocess_json: str
    ) -> None:
        self.config = ConfigTokenizerBertClassification(
            config_data_json,
            config_preprocess_json
        )
        self._load_mecab()
        self._load_bert_tokenizer()
        self.between_sentence_tokens = ['[SEP]']
        return

    def load(self: TokenizerBertClassification) -> None:
        return

    def save(self: TokenizerBertClassification) -> None:
        return

    def preprocess(
        self: TokenizerBertClassification
    ) -> Tuple[TensorDataset, List[Dict]]:
        # read input_json_path and extract documents and labels
        original_list = list()
        parsed_list = list()
        category_list = list()
        for data in self.yield_data_json():
            document = data.get(self.config.document_column)
            category = data.get(self.config.category_column)
            if document is None:
                continue
            if category is None:
                category = ''
            preprocessed = self._preprocess_each_document(document)
            if preprocessed is None:
                continue
            original_list.append(preprocessed['original'])
            parsed_list.append(preprocessed['parsed'])
            category_list.append(category)
        # produce labels by encoding categories
        label_list = self._encode_categories(category_list)
        encs = self.tokenizer.batch_encode_plus(
            parsed_list,
            max_length=self.config.max_word_len+2,
            pad_to_max_length=True
        )
        input_ids = torch.tensor(encs['input_ids']).long()
        token_type_ids = torch.tensor(encs['token_type_ids']).float()
        attention_mask = torch.tensor(encs['attention_mask']).long()
        label_list = torch.tensor(label_list)
        dataset = TensorDataset(
            input_ids, token_type_ids, attention_mask, label_list
        )
        # summarize original data
        resources = list()
        for o, p, c in zip(original_list, parsed_list, category_list):
            resources.append({
                'document': o, 'parsed': p, 'category': c
            })
        return dataset, resources

    def _preprocess_each_document(
        self: TokenizerBertClassification,
        document: str
    ) -> Optional[Dict]:
        document_sentences = self._split_sentence_words(
            self._cleaning_text(document)
        )
        original = ''.join([''.join(s) for s in document_sentences])
        document_sentences = [
            s for s in document_sentences
            if len(s) > self.config.min_word_len
        ]
        if len(document_sentences) < self.config.min_sentences:
            return None
        document_sentences = document_sentences[:self.config.max_sentences]
        document_words = list()
        for s in document_sentences[:-1]:
            document_words.extend(s + self.between_sentence_tokens)
        document_words.extend(document_sentences[-1])
        if len(document_words) >= self.config.max_word_len:
            document_words = document_words[:self.config.max_word_len]
            if document_words[-1] == '[SEP]':
                document_words = document_words[:-1]
        return {
            'original': original,
            'parsed': ' '.join(document_words),
        }
