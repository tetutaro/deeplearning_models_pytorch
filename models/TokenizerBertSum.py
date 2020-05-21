#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional
import os
from itertools import combinations
import torch
from torch.utils.data import TensorDataset
from .Tokenizer import ConfigTokenizer, Tokenizer

DEFAULT_MIN_WORD_LEN = 3
DEFAULT_MAX_WORD_LEN = 510
DEFAULT_MIN_SENTENCES = 3
DEFAULT_MAX_SENTENCES = 100


class ConfigTokenizerBertSum(ConfigTokenizer):
    tokenizer_bertsum_params = [
        # name, vtype, is_require, default
        ('model_name', str, True, None),
        ('document_column', str, True, None),
        ('abstruct_column', str, True, None),
        ('nict_bert_dir', str, True, None),
        ('min_word_len', int, False, DEFAULT_MIN_WORD_LEN),
        ('max_word_len', int, False, DEFAULT_MAX_WORD_LEN),
        ('min_sentences', int, False, DEFAULT_MIN_SENTENCES),
        ('max_sentences', int, False, DEFAULT_MAX_SENTENCES),
    ]

    def __init__(
        self: ConfigTokenizerBertSum,
        config_data_json: str,
        config_preprocess_json: str
    ) -> None:
        # model_name
        config = dict()
        config['model_name'] = 'BertSum'
        self._load_two(config, config_data_json, config_preprocess_json)
        # initialize parent class
        self._init_preprocessor(config)
        self._init_tokenizer(config)
        # set parameters
        for param in self.preprocessor_params:
            self._init_param(config, *param)
        for param in self.tokenizer_params:
            self._init_param(config, *param)
        for param in self.tokenizer_bertsum_params:
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

    def load(self: ConfigTokenizerBertSum) -> None:
        return

    def save(self: ConfigTokenizerBertSum) -> None:
        return


# N-gram を作る
def get_ngrams(n: int, text: List[str]) -> Set:
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


# ROUGEスコア（N-gramが双方に存在するかのF1スコア）を計算する
def calc_rouge(evaluated_ngrams: Set, reference_ngrams: Set) -> float:
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)
    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)
    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count
    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count
    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return f1_score


# 本文の中の文章で、要約の文章とのROUGEスコアが最大になる文章群を選ぶ
def get_oracle_ids(
    document_sentences: List[List[str]],
    abstract_words: List[str],
    summary_size: int
):
    max_rouge = 0
    max_idx = (0, 0)
    evaluated_1grams = [get_ngrams(1, s) for s in document_sentences]
    reference_1grams = get_ngrams(1, abstract_words)
    evaluated_2grams = [get_ngrams(2, s) for s in document_sentences]
    reference_2grams = get_ngrams(2, abstract_words)
    impossible_idxs = []
    for s in range(summary_size):
        combs = combinations(
            [
                i for i in range(len(document_sentences))
                if i not in impossible_idxs
            ],
            s + 1
        )
        for c in combs:
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = calc_rouge(candidates_1, reference_1grams)
            rouge_2 = calc_rouge(candidates_2, reference_2grams)
            rouge_score = rouge_1 + rouge_2
            if (s == 0) and (rouge_score == 0):
                impossible_idxs.append(c[0])
            if rouge_score > max_rouge:
                max_idx = c
                max_rouge = rouge_score
    return sorted(list(max_idx))


class TokenizerBertSum(Tokenizer):
    def __init__(
        self: TokenizerBertSum,
        config_data_json: str,
        config_preprocess_json: str
    ) -> None:
        self.config = ConfigTokenizerBertSum(
            config_data_json,
            config_preprocess_json
        )
        self._load_mecab()
        self._load_bert_tokenizer()
        self.between_sentence_tokens = ['[SEP]', '[CLS]']
        return

    def load(self: TokenizerBertSum) -> None:
        return

    def save(self: TokenizerBertSum) -> None:
        return

    def preprocess(
        self: TokenizerBertSum
    ) -> Tuple[TensorDataset, List[Dict]]:
        # read input_json_path and extract documents and labels
        preprocessed_list = list()
        for data in self.yield_data_json():
            document = data.get(self.config.document_column)
            abstruct = data.get(self.config.category_column)
            if document is None:
                continue
            preprocessed = self._preprocess_each_document(
                document, abstruct
            )
            if preprocessed is None:
                continue
            preprocessed_list.append(preprocessed)
        original_list = [d['original'] for d in preprocessed_list]
        sentences_list = [d['sentences'] for d in preprocessed_list]
        parsed_list = [d['parsed'] for d in preprocessed_list]
        abstruct_list = [d['abstruct'] for d in preprocessed_list]
        segment_ids_list = [d['segment_ids'] for d in preprocessed_list]
        cls_ids_list = [d['cls_ids'] for d in preprocessed_list]
        labels_list = [d['labels'] for d in preprocessed_list]
        # produce labels by encoding categories
        encs = self.tokenizer.batch_encode_plus(
            parsed_list,
            max_length=self.config.max_word_len+2,
            pad_to_max_length=True
        )
        input_ids = torch.tensor(encs['input_ids']).long()
        attention_mask = torch.tensor(encs['attention_mask']).long()
        segment_ids = torch.tensor(
            self._padding_list(segment_ids_list, 0)[1]
        ).long()
        cls_ids = self._padding_list(cls_ids_list, -1)[1]
        cls_ids_mask = [[0 if c == -1 else 1 for c in cs] for cs in cls_ids]
        cls_ids = torch.tensor(cls_ids)
        cls_ids_mask_bool = ~(cls_ids == -1)
        cls_ids[cls_ids == -1] = 0
        cls_ids_mask = torch.tensor(cls_ids_mask).float()
        labels = torch.tensor(
            self._padding_list(labels_list, 0)[1]
        ).float()
        dataset = TensorDataset(
            input_ids, attention_mask, segment_ids,
            cls_ids, cls_ids_mask, cls_ids_mask_bool, labels
        )
        # summarize original data
        resources = list()
        for o, s, p, a in zip(
            original_list, sentences_list, parsed_list, abstruct_list
        ):
            resources.append({
                'document': o, 'sentences': s, 'parsed': p, 'abstruct': a
            })
        return dataset, resources

    def _preprocess_each_document(
        self: TokenizerBertSum,
        document: str,
        abstruct: Optional[str],
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
        sentences = [''.join(s) for s in document_sentences]
        document_words = list()
        for s in document_sentences[:-1]:
            document_words.extend(s + self.between_sentence_tokens)
        document_words.extend(document_sentences[-1])
        if len(document_words) >= self.config.max_word_len:
            document_words = document_words[:self.config.max_word_len]
            if document_words[-1] == '[CLS]':
                document_words = document_words[:-1]
            if document_words[-1] == '[SEP]':
                document_words = document_words[:-1]
        document_words = ['[CLS]'] + document_words + ['[SEP]']
        _segs = [-1] + [
            i for i, w in enumerate(document_words) if w == '[SEP]'
        ]
        segs = [
            _segs[i] - _segs[i - 1] for i in range(1, len(_segs))
        ]
        segment_ids = list()
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segment_ids.extend([0] * s)
            else:
                segment_ids.extend([1] * s)
        cls_ids = [i for i, w in enumerate(document_words) if w == '[CLS]']
        document_words = document_words[1:-1]
        if abstruct is not None:
            abstruct_words = self._split_words(
                self._cleaning_text(abstruct)
            )
            oracle_ids = get_oracle_ids(
                document_sentences, abstruct_words, self.min_sentences
            )
            labels = [0] * len(document_sentences)
            for l in oracle_ids:
                labels[l] = 1
        else:
            labels = [0] * len(document_sentences)
        return {
            'original': original,
            'sentences': sentences,
            'parsed': ' '.join(document_words),
            'abstruct': ''.join(abstruct_words),
            'segment_ids': segment_ids,
            'cls_ids': cls_ids,
            'labels': labels,
        }
