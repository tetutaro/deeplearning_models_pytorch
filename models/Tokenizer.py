#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Generator
import os
from abc import ABC
import subprocess
import simplejson as json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from MeCab import Tagger
from gensim.models.keyedvectors import KeyedVectors
from mojimoji import han_to_zen
from transformers import BertTokenizer
from .Preprocessor import ConfigPreprocessor, Preprocessor

DICTIONARIES = ['ipa', 'juman', 'neologd']
DEFAULT_CHOPPING_WORDS = ["\u3000", " "]
DEFAULT_SEPARATE_WORDS = ["。", "？", "！"]
DEFAULT_EXTRACT_SPEECHES = ["名詞"]
DEFAULT_IS_WAKATI = False
DEFAULT_USE_ORIGINAL = False


class ConfigTokenizer(ConfigPreprocessor, ABC):
    tokenizer_params = [
        # name, vtype, is_require, default
        ('dictionary_name', str, True, None),
        ('keyedvector_bin', str, False, ''),
        ('nict_bert_dir', str, False, ''),
        ('chopping_words', [list, str], False, DEFAULT_CHOPPING_WORDS),
        ('separate_words', [list, str], False, DEFAULT_SEPARATE_WORDS),
        ('extract_speeches', [list, str], False, DEFAULT_EXTRACT_SPEECHES),
        ('is_wakati', bool, False, DEFAULT_IS_WAKATI),
        ('use_original', bool, False, DEFAULT_USE_ORIGINAL),
        ('num_class', int, False, 0),
        ('unique_categories', [list, str], False, []),
    ]

    def _init_tokenizer(
        self: ConfigTokenizer,
        config: Dict
    ) -> None:
        # value assertion
        assert(config['dictionary_name'] in DICTIONARIES)
        return

    def save_categories(
        self: ConfigTokenizer,
        unique_categories: List[str]
    ) -> None:
        self.unique_categories = unique_categories
        self.num_class = len(unique_categories)
        return

    def is_encoded(self: ConfigTokenizer) -> bool:
        return self.num_class > 0


class Tokenizer(Preprocessor, ABC):
    def _load_mecab(self: Tokenizer) -> None:
        mecab_config_path = None
        mecab_config_cands = [
            "/usr/bin/mecab-config", "/usr/local/bin/mecab-config"
        ]
        for c in mecab_config_cands:
            if os.path.exists(c):
                mecab_config_path = c
                break
        assert(mecab_config_path is not None)
        dic_dir = subprocess.run(
            [mecab_config_path, "--dicdir"],
            check=True, stdout=subprocess.PIPE, text=True
        ).stdout.rstrip()
        dic_path = None
        if self.config.dictionary_name == 'ipa':
            dic_cands = ['ipadic-utf8', 'ipadic']
        elif self.config.dictionary_name == 'juman':
            dic_cands = ['juman-utf8', 'jumandic']
        else:  # self.config.dictionary_name == 'neologd'
            dic_cands = ['mecab-ipadic-neologd']
        for c in dic_cands:
            t = os.path.join(dic_dir, c)
            if os.path.isdir(t):
                dic_path = t
                break
        assert(dic_path is not None)
        if self.config.is_wakati:
            self.tagger = Tagger(f"-Owakati -d{dic_path}")
        else:
            self.tagger = Tagger(f"-d{dic_path}")
        if self.config.dictionary_name == 'juman':
            self.orig_index = 4
        else:
            self.orig_index = 6
        return

    def _load_keyedvectors(self: Tokenizer) -> None:
        assert(os.path.exists(self.config['keyedvector_bin']))
        # load KeyedVectors
        self.kvs = KeyedVectors.load_word2vec_format(
            self.config.keyedvector_bin, binary=True
        )
        # save word vectors for using them at the model
        word_vectors = self.kvs.vectors
        word_vectors = np.vstack([
            np.zeros((1, word_vectors.shape[1])), word_vectors
        ])
        setattr(self, 'word_vectors', word_vectors)
        return

    def _load_bert_tokenizer(self: Tokenizer) -> None:
        tokenizer_config_json = os.path.join(
            self.config.nict_bert_dir, 'tokenizer_config.json'
        )
        tokenizer_vocab_path = os.path.join(
            self.config.nict_bert_dir, 'vocab.txt'
        )
        assert(os.path.exists(tokenizer_config_json))
        assert(os.path.exists(tokenizer_vocab_path))
        with open(tokenizer_config_json, 'rt') as rf:
            tokenizer_config = json.load(rf)
        self.tokenizer = BertTokenizer(
            tokenizer_vocab_path, **tokenizer_config
        )
        return

    def _cleaning_text(self: Tokenizer, text: str) -> str:
        text = han_to_zen(text)
        for chopping_word in self.config.chopping_words:
            text = text.replace(chopping_word, '')
        return text

    def _padding_list(
        self: Tokenizer,
        data: List[List[int]],
        padding_id: int,
        max_word_len: Optional(int)
    ) -> Tuple[int, List[List[int]]]:
        if max_word_len is None:
            max_word_len = max(len(d) for d in data)
        ret = [
            d + [padding_id] * (max_word_len - len(d))
            for d in data
        ]
        return max_word_len, ret

    # usage of LabelEncoder
    def _encode_categories(
        self: Tokenizer,
        category_list: List[str]
    ) -> List[int]:
        if self.config.is_encoded():
            category2label = {
                c: i for i, c in enumerate(self.config.unique_categories)
            }
            label_list = [
                category2label.get(c, -1) for c in category_list
            ]
        else:
            cats = np.array(category_list)
            le = LabelEncoder().fit(cats)
            self.config.save_categories(le.classes_.tolist())
            label_list = le.transform(cats).tolist()
        return label_list

    # usage of MeCab (when is_wakati is False)
    def _yield_parsed_node(
        self: Tokenizer,
        text: str
    ) -> Generator[str, None, None]:
        assert(self.config.is_wakati is False)
        nodes = self.tagger.parse(text).strip().split("\n")
        for node in nodes:
            yield node
        return

    # usage of keyedvectors (when is_wakati is False)
    def _get_word_id(
        self: Tokenizer,
        node: str
    ) -> Tuple[Optional[str], Optional[int]]:
        surf_ftrs = node.split("\t")
        if len(surf_ftrs) < 2:
            return (None, None)
        surf = surf_ftrs[0]  # word surface（表層）
        ftrs = surf_ftrs[1].split(",")  # word features
        if len(ftrs) < (self.orig_index + 1):
            return (surf, None)
        spch = ftrs[0]  # part of speech（品詞）
        orig = ftrs[self.orig_index]  # original form（原形）
        if self.config.use_original:
            word = orig
        else:
            word = surf
        if (
            len(self.config.extract_speeches) > 0
        ) and (
            spch not in self.config.extract_speeches
        ):
            return (surf, None)
        vocab = self.kvs.wv.vocab.get(word)
        if vocab is None:
            return (surf, None)
        return (surf, vocab.index + 1)

    def _convert_word_ids(
        self: Tokenizer,
        words: List[str]
    ) -> List[int]:
        word_ids = list()
        for word in words:
            vocab = self.kvs.wv.vocab.get(word)
            if vocab is None:
                continue
            word_ids.append(vocab.index + 1)
        return word_ids

    # usage of MeCab (when is_wakati is True)
    def _split_words(self: Tokenizer, text: str) -> List[str]:
        assert(self.config.is_wakati is True)
        return self.tagger.parse(text).strip().split(' ')

    def _split_sentences(
        self: Tokenizer,
        texts: List[str],
        sep: str
    ) -> List[str]:
        rets = list()
        for t in texts:
            ts = t.split(sep)
            if len(ts) > 1:
                for i in range(len(ts)):
                    ts[i] += sep
            if ts[-1] == '':
                ts = ts[:-1]
            rets.extend(ts)
        return rets

    def _split_sentence_words(
        self: Tokenizer,
        text: str
    ) -> List[List[str]]:
        texts = [text]
        for sep in self.config.separate_words:
            texts = self._split_sentences(texts, sep)
        sentences = list()
        for sentence in texts:
            sentence.append(self._split_words(sentence))
        return sentences
