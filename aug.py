from sklearn.preprocessing import OneHotEncoder
from py_boost import GradientBoosting


import numpy as np
import copy
from sklearn.preprocessing import PolynomialFeatures
from gcforest import *
from data.dataset import *
from sklearn.model_selection import KFold, StratifiedKFold
import random
import ray

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
    def __init__(self, classifier,X_train, y_train, X_test, y_test, gpu_id, num_classes, max_layer=15, aug_type='cutmix',num_df=8):
        self.gpu_id = gpu_id
        self.num_classes=num_classes
        self.classifier = classifier
        self.df_list=[]
        self.max_layer=max_layer
        self.current_layer=0
        self.best_policy=[]
        self.acc_list=[]
        self.policy_list=[]
        self.best_df_index=0
        self.num_df=num_df       
        self.means = np.mean(X_train, axis=0)
        self.stds = np.std(X_train, axis=0)
        self.aug_type = aug_type
        one_hot = OneHotEncoder(sparse_output=False).fit(y_train.reshape(-1,1))
        self.encoder = one_hot
        clf = GradientBoosting('bce', lr=0.3, colsample=0.3, verbose=-1)
        clf.fit(X_train, one_hot.transform(y_train.reshape(-1,1)))
        weights = clf.get_feature_importance()
        weights = [w / sum(weights) for w in weights]
        self.feature_weights = weights
        
        for i in range(self.num_df):   
            df = gcForest(classifier = self.classifier, encoder=self.encoder, num_estimator=100, num_forests=1, max_features=0.2, gpu_id = self.gpu_id, means=self.means, std=self.stds,
                        num_classes=self.num_classes, n_fold=5, max_layer=15, extend=1, weights = self.feature_weights, aug_type = self.aug_type, aug_prob=0.0, aug_mag = 0.0)
            df.load_data(X_train, y_train, X_test, y_test)
            self.df_list.append(df)      
        self.best_policy=self.df_list[0].get_best_policy(X_train, y_train, X_test, y_test)
        #self.best_policy = ['cutmix', 0.7, 0.1]
        #self.best_policy = ['cutmix', 0.3, 0.4]
        #self.best_policy = ['cutmix', 0.5, 0.7]
        #self.best_policy = ['cutmix', 0.7, 0.2]
        self.policy_list.append(self.best_policy)
    '''
    @ray.remote
    def get_best_one(self):
        best_policy=self.df_list[0].get_best_policy(self.df_list[0].current_train_data, self.df_list[0].current_train_label, self.df_list[0].current_test_data, self.df_list[0].current_test_label)
        return best_policy
    '''        
    def generate_policy(self,policy_count):
        generated_policy=generate_combinations(self.best_policy[1],self.best_policy[2],policy_count)
        new_policy=[]
        for i in range(policy_count):
            temp_policy = []
            temp_policy.append(self.aug_type)
            temp_policy.append(generated_policy[i][0])
            temp_policy.append(generated_policy[i][1])
            new_policy.append(temp_policy)
        return new_policy
    
    def renew_policy(self):
        if self.current_layer==0:
            new_policy=self.generate_policy(self.num_df-1)
            self.policy_list.extend(new_policy)
        else:
            new_policy=self.generate_policy(int(self.num_df/2))
            ind = np.argpartition(self.acc_list, int(self.num_df/2))[:int(self.num_df/2)]
            for i in range(int(self.num_df/2)):
                self.policy_list[ind[i]]=new_policy[i]
                #self.df_list[ind[i]]=copy.copy(self.df_list[self.best_df_index])
                self.df_list[ind[i]]=copy.deepcopy(self.df_list[self.best_df_index])
    @ray.remote(num_cpus= 32)
    def train_augDF(df, policy):
        return df.train_by_layer_main(policy)
                    
    def train_augDF_by_layer(self):
        self.renew_policy()
        temp_acc_list=[]
        temp_test_list=[]
        
        tasks = []
        
        for i in range(self.num_df): 
            print("model " + str(i))
            '''
            temp_acc=self.df_list[i].train_by_layer(self.policy_list[i])
            temp_acc_list.append(temp_acc[0])
            temp_test_list.append(temp_acc[1])            
            '''
            #tasks.append(self.df_list[i].train_by_layer(self.policy_list[i]))
            tasks.append(self.train_augDF.remote(self.df_list[i], self.policy_list[i]))
        
        results = ray.get(tasks)
        
        for i, train_result in enumerate(results):
            result = self.df_list[i].train_by_layer_other(self.policy_list[i], train_result)
            temp_acc_list.append(result[0])
            temp_test_list.append(result[1])
        
        self.acc_list=temp_acc_list
        
        #index=len(temp_acc_list)-1-temp_acc_list[::-1].index(max(temp_acc_list))
        #index = len(temp_acc_list)-1-temp_acc_list[::-1].index(min(temp_acc_list))
        max_values = sorted(temp_acc_list, reverse=True)[:2]
        selected_value = random.choice(max_values)
        index = temp_acc_list.index(selected_value)

        self.best_df_index=index
        self.best_policy=self.policy_list[index]
        print("best model index "+str(index))
        print("corrsponding policy "+str(self.best_policy))
        print("best val acc "+str(temp_acc_list[index]))
        print("corrsponding test acc "+str(temp_test_list[index]))
        self.current_layer=self.current_layer+1
        return temp_acc_list, temp_test_list

    def train(self):
        #self.best_policy = ray.get(self.get_best_one.remote(self))
        #self.policy_list.append(self.best_policy)
        for i in range(self.max_layer):
            print("layer " +str(i))
            self.train_augDF_by_layer()
        self.df_list[self.best_df_index].get_best_acc_of_all_layer()
        print(self.df_list[self.best_df_index].aug_policy_schedule)
        return self.df_list[self.best_df_index].aug_policy_schedule
if __name__ == '__main__':    
    ray.init(ignore_reinit_error=True)
    X_train, y_train, X_test, y_test=get_data('adult')
    aug=aug_DF('sketch', X_train, y_train, X_test, y_test, num_classes=2, max_layer=15, gpu_id = 0, aug_type='cutmix')
    aug_policy_schedule = aug.train()
    ray.shutdown()