#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple
from html.parser import HTMLParser
import torch
from torch.utils.data import TensorDataset
from .Tokenizer import ConfigTokenizer, Tokenizer
from .TokenizerTextCNN import ConfigTokenizerTextCNN


class ConfigTokenizerGradCAM(ConfigTokenizer):
    tokenizer_gradcam_params = [
        # name, vtype, is_require, default
        ('model_name', str, True, None),
        ('data_json', str, True, None),
        ('textcnn_config_data_json', str, True, None),
    ]
    textcnn_takeover_params = [
        'data_name',
        'base_dir',
        'keyedvector_bin',
        'num_class',
        'unique_categories',
        'word_len',
    ]

    def __init__(
        self: ConfigTokenizerGradCAM,
        config_data_json: str,
        config_preprocess_json: str
    ) -> None:
        # model_name
        config = dict()
        config['model_name'] = 'GradCAM'
        self._load_one(config, config_data_json)
        # set parameters
        for param in self.tokenizer_gradcam_params:
            self._init_param(config, *param)
        # load ConfigTokenizerTextCNN
        config_tokenizer_textcnn = ConfigTokenizerTextCNN(
            config['textcnn_config_data_json'],
            config_preprocess_json
        )
        config_tokenizer_textcnn.load()
        # take over from textcnn
        for name in self.textcnn_takeover_params:
            setattr(self, name, getattr(config_tokenizer_textcnn, name))
        return

    def load(self: ConfigTokenizerGradCAM) -> None:
        return

    def save(self: ConfigTokenizerGradCAM) -> None:
        return


class ParserGradCAM(HTMLParser):
    def __init__(self: ParserGradCAM) -> None:
        super().__init__()
        self.words = list()
        self.extract = False
        return

    def handle_starttag(self: ParserGradCAM, tag: str, attrs: List) -> None:
        assert(tag == 'span')
        self.extract = True
        return

    def handle_endtag(self: ParserGradCAM, tag: str) -> None:
        assert(tag == 'span')
        assert(self.extract is False)
        return

    def handle_data(self: ParserGradCAM, data: str) -> None:
        if self.extract is True:
            self.words.append(data)
            self.extract = False
        return

    def get_words(self: ParserGradCAM) -> List[str]:
        words = self.words[:]
        self.words = list()
        return words


class TokenizerGradCAM(Tokenizer):
    def __init__(
        self: TokenizerGradCAM,
        config_data_json: str,
        config_preprocess_json: str
    ) -> None:
        self.config = ConfigTokenizerGradCAM(
            config_data_json,
            config_preprocess_json
        )
        self.parser = ParserGradCAM()
        self._load_keyedvectors()
        return

    def load(self: TokenizerGradCAM) -> None:
        return

    def save(self: TokenizerGradCAM) -> None:
        return

    def preprocess(
        self: TokenizerGradCAM
    ) -> Tuple[TensorDataset, List[Dict]]:
        # read input_json_path and extract documents and labels
        document_list = list()
        spanned_list = list()
        word_ids_list = list()
        category_list = list()
        predicted_category_list = list()
        probability_list = list()
        for data in self.yield_data_json():
            document = data.get('document')
            spanned = data.get('spanned')
            category = data.get('category')
            predicted_category = data.get('predicted_category')
            probability = data.get('probability')
            if document is None:
                continue
            if category is None:
                category = ''
            word_ids = self._preprocess_each_document(spanned)
            if len(word_ids) == 0:
                continue
            word_ids_list.append(word_ids)
            document_list.append(document)
            spanned_list.append(spanned)
            category_list.append(category)
            predicted_category_list.append(predicted_category)
            probability_list.append(probability)
        # produce labels by encoding categories
        label_list = self._encode_categories(category_list)
        _, word_ids_list = self._padding_list(
            word_ids_list, 0, self.config.word_len
        )
        word_ids_list = torch.tensor(word_ids_list).long()
        label_list = torch.tensor(label_list)
        dataset = TensorDataset(
            word_ids_list, label_list
        )
        # summarize original data
        documents = list()
        for o, s, c, pred, prob in zip(
            document_list, spanned_list, category_list,
            predicted_category_list, probability_list
        ):
            documents.append({
                'document': o, 'spanned': s, 'category': c,
                'predicted_category': pred,
                'probability': prob,
            })
        return dataset, documents

    def _preprocess_each_document(
        self: TokenizerGradCAM,
        spanned: str
    ) -> List[str]:
        self.parser.feed(spanned)
        words = self.parser.get_words()
        return self._convert_word_ids(words)
