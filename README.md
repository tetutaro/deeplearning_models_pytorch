# deeplearning_models_pytorch

The DeepLearning models which are implemented with [PyTorch](https://pytorch.org/)
and [PyTorch Lightninig](https://github.com/PyTorchLightning/pytorch-lightning).

The text handling models (NLP by DeepLearning) are specifically designed to process Japanese.

[NICT BERT](https://alaginrc.nict.go.jp/nict-bert/index.html) is used as
the pretrained [BERT](https://arxiv.org/abs/1810.04805) model that handles Japanese.

## The DeepLearning models contained in this repository

- TextCNN
    - Kim, Yoon. "Convolutional neural networks for sentence classification." arXiv preprint arXiv:1408.5882 (2014).
    - [arXiv](https://arxiv.org/abs/1408.5882)
- GradCAM
    - Selvaraju, Ramprasaath R., et al. "Grad-cam: Visual explanations from deep networks via gradient-based localization." Proceedings of the IEEE international conference on computer vision. 2017.
    - [arXiv](https://arxiv.org/abs/1609.02391)
    - [Author's original implementation](https://github.com/ramprs/grad-cam)
- BertClassification
    - fine-tuning the BertForSequenceClassification of [HuggingFace](https://huggingface.co/)'s [transformers](https://huggingface.co/transformers/index.html)
    - using [NICT BERT](https://alaginrc.nict.go.jp/nict-bert/index.html)
- BertSum
    - Liu, Yang. "Fine-tune BERT for extractive summarization." arXiv preprint arXiv:1903.10318 (2019).
    - [arXiv](https://arxiv.org/abs/1903.10318)
    - [Author's original implementation](https://github.com/nlpyang/BertSum)
    - using [NICT BERT](https://alaginrc.nict.go.jp/nict-bert/index.html)

## Preparation

- install MeCab and dictonaries for MeCab
    - ex) `brew install mecab mecab-jumandic`
- `pip install -r requirements.txt`
- download `NICT_BERT-base_JapaneseWikipedia_100K.zip` from [NICT BERT](https://alaginrc.nict.go.jp/nict-bert/index.html) site and put it under the `binaries` directory and unzip it.
- create trained word2vec format binary file of [gensim.models.Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html) or [facebook's fastText](https://github.com/facebookresearch/fastText) and put it under the `binaries` directory.
    - ex) `binaries/keyedvectors/fasttext_kv_mecab_juman.bin`
    - cf) [My repsitory for creating trained binary file of Word2Vec/fastText using Japanese Wikipedia](https://github.com/tetutaro/word_embed_binary_jawiki)
- download the Livedoor News Corpus(`ldcc-20140209.tar.gz`) from [Rondhuit's site](http://www.rondhuit.com/download.html#ldcc) and put it under the `data` direcotry and unarchive it.
- `cd data && python jsonize_livedoor_news_corpus.py`
- download `synset_words.txt` of ImageNet Large Scale Visual Recognition Challenge 2012 (ILSVRC2012) and put it under the `data` directory.
- `cd data && python synset_word2labels.py`
- create config JSON files under the `configs` directory.
    - sample of config files are in the `configs/samples` directory
    - ex) `cd configs && ln -s samples/*.json .`

## Structure of this repository

- every deeplearning model has 4 parts
    - Preprocessor: Tokenizer or ImageLoader
    - Model itself
    - Trainer: PyTorch Lightning
    - Processor: organize preprocessing and training, and predict and output the result
- you have to create 5 config files per train/predict
    - data config file
        - ex) configs/samples/data_livedoor_news_corpus.json
    - preprocessing config file
        - ex) configs/samples/tokenizer_textcnn.json
    - model config file
        - ex) configs/samples/model_textcnn.json
    - training config file
        - ex) configs/samples/lightning.json
    - processing config file
        - ex) configs/samples/textcnn_livedoor_news_corpus.json
- you have to specity the processing config file at each run
    - ex) `python textcnn_ja.py -c configs/textcnn_livedoor_news_corpus.json`

## How to use

- TextCNN
    - ex) `python textcnn_ja.py -c configs/textcnn_livedoor_news_corpus.json`
- GradCAM
    - Visualize TextCNN
        - ex) `python gradcam.py -c configs/gradcam_livedoor_news_corpus.json`
    - Visualize ResNet
        - ex) `python gradcam.py -c configs/gradcam_sample_images.json`
- BertClassification
    - ex) `python bert_classification_ja.py -c configs/bert_classificaiton_livedoor_news_corpus.json`
- BertSum
    - ex) `python bertsum_ja.py -c configs/bertsum_livedoor_news_corpus.jon`

- show visualization of TextCNN-GradCAM
    - install nginx and [FooTable](https://fooplugins.github.io/FooTable/)
    - ex) `cd results && python footablize_classfication.py -i GradCAM_TextCNN_livedoor_news_corpus_test.json`
    - start nginx
    - access `http://localhost:8080/GradCAM_TextCNN_livedoor_news_corpus_test`

## Sample images

- `data/sample_images/cat_dog.png`
    - from [GradCAM author's repository](https://github.com/ramprs/grad-cam/blob/master/images/cat_dog.jpg)
- `data/sample_images/bullmastiff.png`
    - from [Wikipeida "Bullmastiff"](https://ja.wikipedia.org/wiki/%E3%83%96%E3%83%AB%E3%83%9E%E3%82%B9%E3%83%86%E3%82%A3%E3%83%95)
- `data/sample_images/egyptiancat.png`
    - from [Wikipedia "Egyptian Mau"](https://en.wikipedia.org/wiki/Egyptian_Mau)
