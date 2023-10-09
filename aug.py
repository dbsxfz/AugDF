from sklearn.preprocessing import OneHotEncoder
from py_boost import GradientBoosting
import sys
import numpy as np
import copy
from sklearn.preprocessing import PolynomialFeatures
from gcforest import *
from data.dataset import get_data
from sklearn.model_selection import KFold, StratifiedKFold
import random
import ray
import argparse
from hyperparameters import *
# need replace eporbs and emags
def generate_combinations(j,k,count):
    index1 = eprobs.index(j)  # 获取第一个列表中指定元素的索引
    index2 = emags.index(k)  # 获取第二个列表中指定元素的索引
    
    combinations = set()  # 存储不同组合的集合
    combinations.add((j, k))  # 将指定的元素组合放入集合中
    while len(combinations) < count+1:
        weights = []
        for j_index in range(len(eprobs)):
            for k_index in range(len(emags)):
                # print(j)
                distance1 = abs(j_index - index1)
                distance2 = abs(k_index - index2)
                weight = 1 / (distance1 + distance2 + 1)  # 计算权重
                weights.append(weight)
        
        # 根据权重选择一个索引对
        random_index = random.choices(range(len(weights)), weights=weights)[0]
        random_index1 = random_index // len(emags)  # 计算第一个列表的索引
        random_index2 = random_index % len(emags)  # 计算第二个列表的索引
        
        combination = (eprobs[random_index1], emags[random_index2])  # 组合元素
        combinations.add(combination)  # 将组合添加到集合中
    
    combinations.remove((j, k))  # 删除输入元素构成的组合
    return list(combinations)


class aug_DF:
    def __init__(self, classifier, gpu_id, num_classes, max_layer=15, extend=1,n_fold=5,aug_type='cutmix',num_forests=1,max_features = 0.3, num_estimator=100,num_df=8,ray=False,random_state=42):
        self.gpu_id = gpu_id
        self.num_classes=num_classes
        self.classifier = classifier
        self.df_list=[]
        self.max_layer=max_layer
        self.current_layer=0
        self.best_policy=[]
        self.acc_list=[]
        self.policy_list=[]
        self.extend=extend
        self.best_df_index=0
        self.num_df=num_df       
        self.aug_type = aug_type
        self.ray=ray
        self.n_fold=n_fold
        self.num_forests = num_forests
        self.max_features = max_features
        self.num_estimator = num_estimator
        self.random_state=random_state
        if self.aug_type=='cutmix' or 'erase':
            self.aug_prob_list = eprobs
            self.aug_mag_list = emags
     
    def generate_policy(self,policy_count):
        index1 = self.aug_prob_list.index(self.best_policy[1])  # 获取第一个列表中指定元素的索引
        index2 = self.aug_mag_list.index(self.best_policy[2])  # 获取第二个列表中指定元素的索引
        
        combinations = set()  # 存储不同组合的集合
        combinations.add((self.aug_type,self.best_policy[1], self.best_policy[2]))  # 将指定的元素组合放入集合中
        while len(combinations) < policy_count+1:
            weights = []
            for j_index in range(len(self.aug_prob_list)):
                for k_index in range(len(self.aug_mag_list)):
                    # print(j)
                    distance1 = abs(j_index - index1)
                    distance2 = abs(k_index - index2)
                    weight = 1 / (distance1 + distance2 + 1)  # 计算权重
                    weights.append(weight)
            
            # 根据权重选择一个索引对
            random_index = random.choices(range(len(weights)), weights=weights)[0]
            random_index1 = random_index // len(self.aug_mag_list)  # 计算第一个列表的索引
            random_index2 = random_index % len(self.aug_mag_list)  # 计算第二个列表的索引
            
            combination = (self.aug_type,self.aug_prob_list[random_index1], self.aug_mag_list[random_index2])  # 组合元素
            combinations.add(combination)  # 将组合添加到集合中
        
        combinations.remove((self.aug_type,self.best_policy[1], self.best_policy[2]))  # 删除输入元素构成的组合
        print(combinations)
        return list(combinations)
    
    def renew_policy(self):
        if self.current_layer==0:
            new_policy=self.generate_policy(self.num_df-1)
            self.policy_list.extend(new_policy)
        else:
            new_policy=self.generate_policy(int(self.num_df/2))
            ind = np.argpartition(self.acc_list, int(self.num_df/2))[:int(self.num_df/2)]
            for i in range(int(self.num_df/2)):
                self.policy_list[ind[i]]=new_policy[i]
                self.df_list[ind[i]]=copy.deepcopy(self.df_list[self.best_df_index])
                
    @ray.remote
    def fit_one_layer_ray(self,df, policy):
        return df.fit_one_layer(policy)
    
    def fit_one_layer(self,df,policy):
        return df.fit_one_layer(policy)    
                
    def train_augDF_by_layer(self):
        self.renew_policy()
        temp_acc_list=[]
        tasks = []
        for i in range(self.num_df): 
            print("model " + str(i))
            if self.ray==True:
                tasks.append(self.fit_one_layer_ray.remote(self.df_list[i], self.policy_list[i]))
            elif self.ray==False:
                tasks.append(self.fit_one_layer(self.df_list[i], self.policy_list[i]))
        if self.ray==True:
            results = ray.get(tasks)
        elif self.ray==False:
            results = tasks
        temp_acc_list=results
        print(results)
        self.acc_list=temp_acc_list
        max_values = sorted(temp_acc_list, reverse=True)[:2]
        selected_value = random.choice(max_values)
        index = temp_acc_list.index(selected_value)

        self.best_df_index=index
        self.best_policy=self.policy_list[index]
        print("best model index "+str(index))
        print("corrsponding policy "+str(self.best_policy))
        print("best val acc "+str(temp_acc_list[index]))
        self.current_layer=self.current_layer+1
        return temp_acc_list

    def train(self):
        for i in range(self.max_layer):
            print("layer " +str(i))
            self.train_augDF_by_layer()
        self.df_list[self.best_df_index].get_best_acc_of_all_layer()
        print(self.df_list[self.best_df_index].aug_policy_schedule)
        return self.df_list[self.best_df_index].aug_policy_schedule
    def fit(self,X_train, y_train):
        self.means = np.mean(X_train, axis=0)
        self.stds = np.std(X_train, axis=0)
        one_hot = OneHotEncoder(sparse_output=False).fit(y_train.reshape(-1,1))
        self.encoder = one_hot
        clf = GradientBoosting('bce', lr=0.3, colsample=0.3, verbose=-1)
        clf.fit(X_train, one_hot.transform(y_train.reshape(-1,1)))
        weights = clf.get_feature_importance()
        weights = [w / sum(weights) for w in weights]
        self.feature_weights = weights
        
        for i in range(self.num_df):   
            df = gcForest(classifier = self.classifier, encoder=self.encoder, num_estimator=self.num_estimator, num_forests=self.num_forests, max_features=self.max_features, gpu_id = self.gpu_id, means=self.means, std=self.stds,
                        num_classes=self.num_classes, n_fold=self.n_fold, max_layer=self.max_layer, extend=self.extend, weights = self.feature_weights,random_state=self.random_state, aug_type = self.aug_type, aug_prob=self.aug_prob_list, aug_mag = self.aug_mag_list,ray=self.ray)
            df.load_data(X_train, y_train)
            self.df_list.append(df)      
        self.best_policy=self.df_list[0].get_best_policy(X_train, y_train)
        self.policy_list.append(self.best_policy)
        return self.train()
    
    # output the best model's predict proba
    def predict_proba(self,test_data):
        return self.df_list[self.best_df_index].predict_proba(test_data)
    def predict(self,test_data):
        return self.df_list[self.best_df_index].predict(test_data)
    def score(self,test_data,test_label):
        return self.df_list[self.best_df_index].score(test_data,test_label)
    def get_searched_policy(self):
        return self.df_list[self.best_df_index].search_schedule
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Customize model configuration")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--aug_type', type=str, default='cutmix', help='Data augmentation type')
    parser.add_argument('--random_state', type=int, default=42, help='Random state')
    parser.add_argument('--use_ray', type=int, default=0, help='Use Ray (0 or 1)')
    parser.add_argument('--max_features', type=float, default=0.3, help='Max features')
    parser.add_argument('--max_layer', type=int, default=15, help='Max layer depth')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--classifier', type=str, default='sketch', help='Base classifier')
    parser.add_argument('--num_forests', type=int, default=1, help='Number of forests')
    parser.add_argument('--num_estimator', type=int, default=100, help='Number of estimators')
    args = parser.parse_args()   
    if args.use_ray:
        ray.init(ignore_reinit_error=True)
    X_train, y_train, X_test, y_test=get_data(args.dataset)
    num_classes = int(np.max(y_train) + 1)
    aug=aug_DF(classifier=args.classifier,num_classes=num_classes,\
        max_layer=args.max_layer, gpu_id = args.gpu_id, aug_type=args.aug_type,ray=args.use_ray,\
        max_features=args.max_features,random_state=args.random_state,\
        num_forests=args.num_forests,num_estimator=args.num_estimator)
    aug_policy_schedule = aug.fit(X_train, y_train)
    if args.use_ray:
        ray.shutdown()
    print(aug_policy_schedule)
    print(aug.predict_proba(X_test))
    aug.score(X_test, y_test)