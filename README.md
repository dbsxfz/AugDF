# AugDF
## Introduction
This repository hosts the code for the paper titled 'Improve Deep Forest with Learnable Layerwise Augmentation Policy Schedule', submitted to ICASSP 2024.

Arxiv preprint version is available at https://arxiv.org/abs/2309.09030v1.

The core implementation is now accessible, with the repository currently undergoing reorganization for enhanced clarity and usability.

Further updates, including trials of parallel computation for acceleration and comparative experiments, will be released soon.
## Usage
To perform policy schedule learning or directly employ the learned schedules, you could utilize the `aug.py` file and the `gcforest.py` file, respectively. Please refer to the `run.sh` script for specific commands and parameter configurations. Detailed instructions for using each function can be found in the `example.ipynb` file. These instructions provide APIs similar to those of sklearn, including functions like `fit`, `predict`, and `predict_proba`.
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

## Hyper-Parameters Settings
### Time Limit for AutoGluon-Tabular
A time constraint is set for AutoGluon-Tabular models training: 30 minutes for datasets with less than 100k samples, and 60 minutes for larger datasets. The computation is performed on a single RTX 4090 GPU.

### Model Capacity
A lightweight configuration is utilized for AugDF, employing one forest per layer. Each forest comprises 100 trees in binary classification tasks, and 150 trees in multi-class classification tasks. The configurations for other Deep Forest variants are similar. For a fair comparison between deep forests and shallow tree ensembles, and to fully test the potential of shallow decision forests, all shallow tree ensembles (including RF, XGBoost, LightGBM, and CatBoost) are equipped with 1,000 trees.

### Hyper-Parameters tuning
Hyper-Parameter tuning is crucial for the performance of decision forests, especially GBDTs. In this work, a grid search is conducted for tuning the 'learning_rate', 'max_depth' for XGBoost, and 'learning_rate', 'num_leaves' for LightGBM. For catboost, automatic learning rate adaptation is performed with its inherent functions. For all deep forests, no hyper-parameter tuning was conducted. The rationale behind this in the vanilla deep forest is primarily attributed to the stability of its base learner (RF). Slightly tuning the GBDTs in the first layer of AugDF might further boost its task-specific generalization ability, though the universal parameters used in our implementation already demonstrate high performance. 

It is noteworthy that the parameter 'max_features' plays a critical role and can be particularly challenging to fine-tune. Enhanced features (validation probabilities derived from the preceding layer), which often possess higher-level semantics and are frequently associated with greater feature importance, tend to result in substantial information gain or Gini purity gain when used for splitting. This propensity may, however, lead to a loss of diversity in the base learners (decision trees) in subsequent ensemble layers, thereby exacerbating the risk of overfitting. Consequently, a judicious approach to column subsampling is generally essential. However, excessive column subsampling could lead to underfitting at the individual layer level and the accumulation of bias, making the determination of an optimal 'max_features' for the entire Deep Forest highly non-trivial. At present, we provisionally set 'max_features' to 0.3 for multi-class problems and to 0.2 for binary classification problems (0.3 if the number of original features is fewer than 10). The comprehensive fine-tuning of hyperparameters for AugDF is considered an avenue for future work, and we would greatly appreciate any suggestions or assistance in this regard.

The vanilla Deep Forest employs Random Forest as its base learners, which traditionally use a low default column subsampling rate (often set to sqrt(n), n is the number of features). This has been widely validated to effectively mitigate the issues of overfitting and loss of diversity among base learners. Unfortunately, popular implementations of Gradient Boosting Decision Trees (GBDT), such as XGBoost, LightGBM, and CatBoost, do not set column subsampling by default due to the much shallower trees in boosting algorithms. Using these GBDT algorithms directly within the Deep Forest framework tends to result in severe overfitting. We speculate that this is one reason why Deep Forest and GBDT technologies have developed somewhat orthogonally, with scant literature successfully employing GBDT as base learners in Deep Forests. A more formal analysis of this phenomenon may appear in an extended version of our paper and is considered a topic for future work. Should the initial insights outlined in this document prove valuable to your research or implementation, we would be deeply honored if you could cite both the current version of our paper and this repository.






