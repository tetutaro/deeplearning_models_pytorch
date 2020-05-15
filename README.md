# deeplearning_models_pytorch
PyTorch + PyTorch Linghtning で作った DeepLearning のモデル集

## 動かし方

- textcnn_ja.py: 日本語のTextCNN
- gradcam.py: GradCAM
- bertclassification_ja.py: 日本語のBertForSequenceClassification
- bertsum_ja.py: 日本語のBertSum（Bert部分の重み更新無し）

## 実装メモ

最下層のclassしか扱わない
- FromConfig(Enum): 読み込んだ設定JSONのkey名
- Config(object)
        - `init_param()`: 継承した`__init__()`内で値のセット
        - `load_one()`: ひとつのJSONを読む
        - `load_two()`: ふたつのJSONを読む
        - `save()`: JSONに書き出す
        - `save_param()`: 継承した`save()`内で値を取り込む
    - ConfigPreprocess(Config): PreprocessのConfig
            - `__init__()`: 最下層から来た`model_name`からpathを作る
            - `setup()`: `data_json`から`data_name`をparseする
            - `config_json` = binaries/`model_name`/`data_name`/tokenizer_config.json
        - ConfigTokenizer(ConfigPreprocess)
            - ConfigTokenizerTextCNN(ConfigTokenizer)
                    - `model_name` = TextCNN
            - ConfigTokenizerGradCAM(ConfigTokenizer)
            - ConfigTokenizerBertClassification(ConfigTokenizer)
            - ConfigTokenizerBertSum(ConfigTokenizer)
        - ConfigImageLoader(ConfigPreprocess)
            - ConfigImageLoaderGradCAM(ConfigPreprocess)
    - ConfigDeepLearning(Config)
            - `data_name`: tokenizerから引き継ぎ
            - `config_json` = binaries/`model_name`/`data_name`/config.json
            - `binary_path` = binaries/`model_name`/`data_name`/pytorch_model.bin
        - ConfigTextCNN(ConfigDeepLearning)
                - `model_name` = TextCNN
        - ConfigGradCAM(ConfigDeepLearning)
        - ConfigBertClassificaton(ConfigDeepLearning)
        - ConfigBertSum(ConfigDeepLearning)
    - ConfigLightning(Config): LightningのConfig
    - ConfigProcess(Config): ProcessのConfig
        - ConfigProcessTextCNN(ConfigDeepLearning)
        - ConfigProcessGradCAM(ConfigDeepLearning)
        - ConfigProcessBertClassificaton(ConfigDeepLearning)
        - ConfigProcessBertSum(ConfigDeepLearning)

- Preprocess(object)
        - `yield_data_json()`: data_jsonのyield
    - Tokenizer(Preprocess): MeCab, KeyedVector, LabelEncoderを扱う
            - `load_mecab()`: MeCab.Taggerのload
            - `load_keyedvectors()`: gemsim.KeyedVectorsのLoad
            - `encode_categories()`: LabelEncoder
            - `get_words()`: MeCabのwakati
            - `yield_parsed_node()`: MeCabのparse
            - `get_word_id()`: KeyedVectorのword index
        - TokenizerTextCNN(Tokenizer)
                - `preprocess()`: TensorDatasetを作る
        - TokenizerGradCAM(Tokenizer)
        - TokenizerBertClassification(Tokenizer):
        - TokenizerBertSum(Tokenizer)
    - ImageLoader(Preprocess): 画像を読み込む
        - ImageLoaderGradCAM(ImageLoader)

- DeepLearning(nn.Module)
    - TextCNN(Model): TextCNNをするnn.Module
            - `__init__()`: そのまま
            - `forward()`: そのまま
            - `load()` = `load_state_dict()`
            - `save()` = `save_state_dict()`
    - GradCAM(Model): GradCAMをするnn.Module
    - BertClassification(Model)
    - BertSum(Model)

- Lightning(LightningModule)
        - `prepare_data()`
        - `train_dataloader()`
        - `val_dataloader()`
        - `configure_optimizer()`
        - `fit()`: Trainer.fit()
    - LightningTextCNN(Lightning)
            - `training_step()`
            - `training_epoch_end()`
            - `validation_step()`
            - `validation_epoch_end()`
    - LightningBertClassification(Lightning)
    - LightningBertSum(Lightning)

- Process(object)
    - ProcessTextCNN(Process):
    - ProcessGradCAM(Process):
    - ProcessBertClassification(Process):
    - ProcessBertSum(Process):
