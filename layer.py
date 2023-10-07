from py_boost import GradientBoosting
from sklearn.model_selection import KFold, StratifiedKFold

import math
import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle
import random
from sklearn.metrics import accuracy_score
from copy import deepcopy

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def random_erase(data, means, prob, mag, k, label, layer_index):#, predictor
    rand_prob = np.random.random(data.shape[0])

    selected_rows_idx = np.where(rand_prob < prob)
    not_selected_rows_idx = np.where(rand_prob >= prob)
    
    save_to_be_perturbed = data[selected_rows_idx]
    not_perturbed = data[not_selected_rows_idx]
    not_perturbed_labels = label[not_selected_rows_idx]

    rand_mag = np.random.random(data[selected_rows_idx, :k].shape)

    data[selected_rows_idx, :k] *= (rand_mag >= mag)
    
    for i in range(1):
        rand_mag_ = np.random.random(data[selected_rows_idx, :k].shape)
        save_to_be_perturbed[:,:k] *= (rand_mag_.squeeze() >= mag)
        perturbed = save_to_be_perturbed.squeeze()
        data = np.vstack((data, perturbed))
    
    return data, selected_rows_idx

def cutmix(data, means, prob, mag, k, label, layer_index, num_classes, weights):#, predictor
    
    class_indices = np.argmax(label, axis=1)
    sample_indices = [[] for _ in range(num_classes)]
    #sample_indices = [[]] * num_classes # wrong but good
    for i, class_index in enumerate(class_indices):
        sample_indices[class_index].append(i)
    
    # 对每条数据以prob为概率做扰动
    rand_prob = np.random.random(data.shape[0])
    # 选取符合概率条件的行
    selected_rows_idx = np.where(rand_prob < prob)
    num_selected = selected_rows_idx[0].shape[0]
    
    mixed_index = [0] * num_selected
    for i, row_idx in enumerate(selected_rows_idx[0]):
        mixed = random.choice([j for j in range(num_classes) if j != class_indices[row_idx]])
        mixed_index[i] = random.choice(sample_indices[mixed])
    
    save_to_be_perturbed = data[selected_rows_idx]
    y_extend = np.zeros((num_selected, num_classes))

    for iter in range(1):
        # 从beta分布中采样得到lamda, 随机选lam*原始特征数个特征列做增强
        lam = np.random.beta(mag, mag)
        target = list(range(data.shape[1]))
        #target = list(range(k))
        random.shuffle(target)
        # 让选取的特征数最少为原始特征数的1 / 4，防止追加原样本或与原样本过于类似
        #tmp = np.maximum(1, (int)(lam * data.shape[1]))
        if k < 100:
            tmp = np.maximum((int)(k / 5), (int)(lam * k))
        else:
            tmp = np.maximum((int)(k / 20), (int)(lam * k))
        #tmp = (int)(lam * k)
        
        weights = np.nan_to_num(weights, nan=0.0)
        added = [sum(weights) / len(weights)] * (data.shape[1] - len(weights))
        if layer_index == 1:
            weights = np.append(weights, added)
            
        weights_scal = [w / sum(weights) for w in weights]
        weight_sum = 0
        for i in target[:tmp]:
            weight_sum += weights_scal[i]

        for i, index in enumerate(mixed_index):
            save_to_be_perturbed[i, target[:tmp]] = data[index, target[:tmp]]
        
        perturbed = save_to_be_perturbed.squeeze()
        y_extend = (1 - weight_sum) * label[selected_rows_idx] + weight_sum * label[mixed_index]
        data = np.vstack((data, perturbed))
    
    return data, y_extend

def compute_accuracy(label, predict):
    if len(predict.shape) > 1:
        test = np.argmax(predict, axis=1)
    else:
        test = predict
        
    if len(label.shape) > 1:
        label = np.argmax(label, axis=1)
    
    test_copy = test.astype("int")
    label_copy = label.astype("int").ravel()
    acc = np.sum(test_copy == label_copy) * 1.0 / len(label_copy) * 100
    return acc

# 拿到一份训练数据，并显式额外拿到一份验证数据，自己不做任何划分
# num_forests与fold无关，一个森林（可能内部是好几个）会占一组val_concatenate的格子
# 返回用来计算验证准确率的平均单层输出（forests的平均）以及用来占格子的未平均验证输出（只对fold平均，不对forest平均）
def train_one_layer(self, train_data, train_label, val_data):
    layer = []
    weight = np.zeros([self.num_forests, train_data.shape[1]])
    val_prob = np.zeros([self.num_forests, val_data.shape[0], self.num_classes])
    
    if self.current_layer_index < 5:
        learning_rate = 0.3
    elif self.current_layer_index >= 5 and self.current_layer_index < 10:
        learning_rate = 0.25
    else:
        learning_rate = 0.2
    for forest_index in range(self.num_forests):
        clf = GradientBoosting('bce', ntrees=self.num_estimator, lr=learning_rate, colsample=self.max_features, verbose=-1)
        clf.fit(train_data, train_label)
        weight[forest_index] = clf.get_feature_importance()
        val_prob[forest_index, :] = clf.predict(val_data)
        layer.append(clf)
        
    val_avg = np.sum(val_prob, axis=0) / self.num_forests
    mean_weight = np.mean(weight, axis=0)
    val_concatenate = val_prob.transpose((1, 0, 2))
    val_concatenate =val_concatenate.reshape(val_concatenate.shape[0], -1)
    return [val_avg, val_concatenate, layer, mean_weight]

def predict_one_layer(self, layer, test_data):
    predict_prob = np.zeros([self.num_forests, test_data.shape[0], self.num_classes])
    
    for forest_index, clf in enumerate(layer):
        predict_prob[forest_index, :] = clf.predict(test_data)
    
    predict_avg = np.sum(predict_prob, axis=0) / self.num_forests
    predict_concatenate = predict_prob.transpose((1, 0, 2))
    predict_concatenate = predict_concatenate.reshape(predict_concatenate.shape[0], -1)
    return [predict_avg, predict_concatenate]

# 对于每一层实际拿到的数据，层内划分fold
# 返回的分别是层内五折交叉验证结果，以及用于占位的概率列，均与输入数据行数相同
def train_KfoldWarpper(self, train_data, train_label, prob, mag, kf):
    KfoldWarpper = []
    mean_weights = np.empty([self.n_fold, train_data.shape[1]])
    num_samples = train_data.shape[0]
    val_prob = np.empty([num_samples, self.num_classes])
    val_prob_concatenate = np.empty([num_samples, self.num_forests * self.num_classes])
    i = 0
    for train_index, test_index in kf.split(train_data, train_label):
        train_label_onehot = self.encoder.transform(train_label.reshape(-1,1))

        y_train = train_label_onehot[train_index]
        y_train = np.tile(y_train, self.extend)
        
        X_train = train_data[train_index, :]
        X_train = np.tile(X_train, (self.extend, 1))
        
        X_val = train_data[test_index, :]
        
        if prob > 0 and mag > 0:
            # no implementation of random_erase
            if self.aug_type == 'erase':
                X_train, idx = random_erase(deepcopy(X_train), means=None, prob=prob, mag=mag, k=self.num_features, 
                            label=deepcopy(y_train), layer_index=self.current_layer_index)#, predictor=self.predict_df
                y_extend = y_train[idx]

                for i in range(1):
                    y_train = np.concatenate((y_train, y_extend),axis=0)
            elif self.aug_type == 'cutmix':
                # print(self.feature_weights)
                X_train, y_extend = cutmix(deepcopy(X_train), means=None, prob=prob, mag=mag, k=self.num_features, 
                            label=deepcopy(y_train), layer_index=self.current_layer_index, num_classes = self.num_classes, weights=self.feature_weights)#, predictor=self.predict_df
                
                for i in range(1):
                    y_train = np.concatenate((y_train, y_extend),axis=0)    
        val_prob[test_index], val_prob_concatenate[test_index, :], layer, mean_weights[i] = train_one_layer(self, X_train, y_train, X_val)
        i = i + 1
        KfoldWarpper.append(layer)
    mean_weight = np.mean(mean_weights, axis = 0)
    return val_prob, val_prob_concatenate, KfoldWarpper, mean_weight

# 对test得到输出与概率列，不同的是所有数据的预测均由所有fold的森林共同给出
def predict_KfoldWarpper(self, KfoldWarpper, test_data):
    test_prob = np.zeros([test_data.shape[0], self.num_classes])
    test_prob_concatenate = np.zeros([test_data.shape[0], self.num_forests * self.num_classes])

    for layer in KfoldWarpper:
        temp_prob, temp_prob_concatenate = predict_one_layer(self, layer, test_data)
        test_prob += temp_prob
        test_prob_concatenate += temp_prob_concatenate
        
    test_prob /= self.n_fold
    test_prob_concatenate /= self.n_fold

    return [test_prob, test_prob_concatenate]