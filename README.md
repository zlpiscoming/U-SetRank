# U-SetRank

> Tenforflow implementation of [CCIR 2022] U-SetRank：基于注意力机制修正的 SetRank 纠偏。
[![Python 3.6](https://img.shields.io/badge/python-3.6%20%7C%203.7-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellowgreen.svg)](https://opensource.org/licenses/Apache-2.0)

## Usage

### Environmental preparation
> you should first download galago and put it in \scripts.

> tf 1.0 and tensorboardX are also used


### Quick Start
> If you have downloaded **Istella Dataset** please comment `sh download_data.sh`.
```bash
sh test.sh
```
### Data Preparation
```bash
cd data
sh download_data.sh
python norm_split_dataset.py
```

### Model Training/Testing
```bash
sh ./scripts/train_lambdamart_istella.sh
sh ./scripts/prepare_data_lambda_istella.sh
sh ./scripts/train_transformer_istella.sh
```

# Datasets
## Istella Dataset
More information in https://tensorflow.google.cn/datasets/catalog/istella


## License

[Apache-2.0](https://opensource.org/licenses/Apache-2.0)

