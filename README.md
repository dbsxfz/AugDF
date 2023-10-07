# AugDF
## Introduction
This repository hosts the code for the paper titled 'Improve Deep Forest with Learnable Layerwise Augmentation Policy Schedule', submitted to ICASSP 2024.

Arxiv preprint version is available at https://arxiv.org/abs/2309.09030v1.

The core implementation is now accessible, with the repository currently undergoing reorganization for enhanced clarity and usability.

Further updates, including trials of parallel computation for acceleration and comparative experiments, will be released soon.

## Dependencies
The core libraries utilized in our implementation are outlined below. Some of them, i.e., deep-forest and autogluon-tabular, are used solely for benchmarking (irrelevant to the implemention of AugDF). We recommend the use of multiple virtual environments to manage potential conflicts seamlessly.

| Library             | Version |
|---------------------|---------|
| Python              | 3.9.17  |
| scikit-learn        | 1.2.2   |
| xgboost             | 1.7.6   |
| lightgbm            | 4.0.0   |
| catboost            | 1.2     |
| deep-forest         | 0.1.7   |
| py-boost            | 0.4.3   |
| autogluon-tabular   | 0.8.2   |

## Hyper-Parameters
# Time Limit for AutoGluon-Tabular
A time constraint is set for AutoGluon-Tabular models training: 30 minutes for datasets with less than 100k samples, and 60 minutes for larger datasets. The computation is performed on a single RTX 4090 GPU.

# Model Capacity
A lightweight configuration is utilized for AugDF, employing one forest per layer. Each forest comprises 100 trees in binary classification tasks, and 150 trees in multi-class classification tasks. The configurations for other Deep Forest variants are similar. For a fair comparison between deep forests and shallow tree ensembles, and to fully test the potential of shallow decision forests, all shallow tree ensembles (including RF, XGBoost, LightGBM, and CatBoost) are equipped with 1,000 trees.

# Hyper-Parameters tuning:
Hyper-Parameter tuning is crucial for the performance of decision forests, especially GBDTs. In this work, a grid search is conducted for tuning the 'learning_rate', 'max_depth' for XGBoost, and 'learning_rate', 'num_leaves' for LightGBM. For catboost, automatic learning rate adaptation is performed with its inherent functions. For all deep forests, oo hyper-parameter tuning was conducted. The rationale behind this in the vanilla deep forest is primarily attributed to the stability of its base learner (RF). Slightly tuning the GBDTs in the first layer of AugDF might further boost its task-specific generalization ability, though the universal parameters used in our implementation already demonstrate high performance.
