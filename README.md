# AugDF
## Introduction
This repository hosts the code for the paper titled 'Improving Deep Forest with Learnable Layerwise Augmentation Policy Schedule', submitted to ICASSP 2024.

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
