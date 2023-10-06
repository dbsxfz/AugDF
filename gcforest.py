from layer import *
#from layer_light_rand import *
import ray

#types = ['erase', 'noise']
eprobs = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 1.0]
emags = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]#0.05, 
nprobs = [0.0, 0.05, 0.1, 0.2, 0.3]
nmags = [0.05, 0.1, 0.2, 0.3, 0.5]

class gcForest:
    def __init__(self, classifier, num_estimator, encoder, num_forests, num_classes, means, std, max_layer=100, 
                max_depth=31, max_features=0.1, n_fold=5, min_samples_leaf=1, sample_weight=None, gpu_id=3, random_state=42, weights=[None],
                purity_function="gini" , bootstrap=True, parallel=True, num_threads=-1, extend=1, aug_type='erase', aug_prob=0.0, aug_mag = 0.0 ):
        self.gpu_id = gpu_id
        self.classifier = classifier
        self.encoder = encoder
        self.num_estimator = num_estimator
        self.num_forests = num_forests
        self.num_classes = num_classes
        self.n_fold = n_fold
        self.max_depth = max_depth
        self.max_features = max_features
        self.max_layer = max_layer 
        self.min_samples_leaf = min_samples_leaf
        self.sample_weight = sample_weight
        self.random_state = random_state
        self.purity_function = purity_function
        self.bootstrap = bootstrap
        self.parallel = parallel
        self.num_threads = num_threads
        self.feature_weights = weights
        
        self.means =means
        self.std=std
        self.extend = extend
        
        self.aug_type = aug_type
        self.aug_prob = aug_prob
        self.aug_mag = aug_mag

        self.model = []
        self.current_layer_index = 0
        self.current_ensemble_prob=0
        self.best_layer_index = 0
        self.best_val_acc = 0
        self.best_test_acc = 0
        
        self.kf_N = 5
        self.kf_val = StratifiedKFold( self.kf_N, shuffle=True, random_state=42)
        
        self.val_index_list = []
        self.train_index_list = []

        self.aug_policy_schedule = []
                      
    def load_data(self, train_data, train_label, X_test, y_test):
        
        num_classes = int(np.max(train_label) + 1)
        if( num_classes != self.num_classes ):
            raise Exception("init num_classes not equal to actual num_classes")
        
        self.current_train_data = train_data
        self.current_train_label = train_label
        self.current_test_data = X_test
        self.current_test_label = y_test
            
        self.back_train_data = train_data
        self.back_train_label = train_label
        self.back_test_data = X_test
        self.back_test_label = y_test
        
        self.num_features = train_data.shape[1]
        
        for train_index, val_index in self.kf_val.split(train_data, train_label):
            self.val_index_list.append(val_index)
            self.train_index_list.append(train_index)
            
    @ray.remote
    def try_one_time(self, train_data, train_label, test_data, test_label, prob, mag, kf):
        # train_KfoldWarpper内部已经做了五折交叉，当我们不考虑外五折时，它的输出就是验证结果

        val_prob, val_stack, layer, mean_weight = train_KfoldWarpper(self, train_data, train_label, prob, mag, kf)
        test_prob, test_stack = predict_KfoldWarpper(self, layer, test_data)
        
        temp_val_acc = compute_accuracy(train_label, val_prob)
        temp_test_acc = compute_accuracy(test_label, test_prob)
        return temp_val_acc, temp_test_acc
    
    def try_one_time_all(self, train_data, train_label, test_data, test_label, prob, mag, kf):
        # train_KfoldWarpper内部已经做了五折交叉，当我们不考虑外五折时，它的输出就是验证结果
        
        val_prob, val_stack, layer, mean_weight = train_KfoldWarpper(self, train_data, train_label, prob, mag, kf)
        test_prob, test_stack = predict_KfoldWarpper(self, layer, test_data)
        
        temp_val_acc = compute_accuracy(train_label, val_prob)
        temp_test_acc = compute_accuracy(test_label, test_prob)
        return [temp_val_acc, temp_test_acc, val_prob, val_stack, test_prob, test_stack, layer, mean_weight]

    def get_best_policy(self,train_data, train_label, X_test, y_test):

        layer_index = self.current_layer_index
        kf = StratifiedKFold( self.n_fold, shuffle=True, random_state = self.random_state + self.current_layer_index)
        print("\n--------------\nlayer {},   X_train shape:{}, X_test shape:{}...\n ".format(str(layer_index), self.current_train_data.shape, self.current_test_data.shape) )

        best_rand_val_acc = 0.0
        best_rand_policy = []
        X_train = self.current_train_data
        y_train = self.current_train_label
        X_test = self.current_test_data
        y_test = self.current_test_label                
        temp = []  
        result_set = [(j, k) for j in eprobs for k in emags if k >= j + 0.1]

        for (j,k) in result_set:
                temp.append(self.try_one_time.remote(self, X_train, y_train, X_test, y_test, prob=j, mag=k, kf=kf))
                
        temp_result = ray.get(temp)      
        for i, temp in enumerate(temp_result):      
            rand_val_acc = temp[0]
            rand_test_acc = temp[1]
            j = result_set[i][0]
            k = result_set[i][1]
            print(self.aug_type,' ',j,' ',k,' ', rand_val_acc,' ',rand_test_acc,' ')
            if rand_val_acc >= best_rand_val_acc:
                best_rand_val_acc = rand_val_acc
                best_rand_policy = [self.aug_type, j, k]
        
        print('final best policy, ', best_rand_policy)
        return best_rand_policy       
    
    def train_by_layer_main(self, best_policy):
        
        kf = StratifiedKFold( self.n_fold, shuffle=True, random_state=self.random_state + self.current_layer_index)
        return self.try_one_time_all(self.current_train_data, self.current_train_label, self.current_test_data, self.current_test_label, prob=best_policy[1], mag=best_policy[2], kf=kf)
    
    def train_by_layer_other(self, best_policy, train_result):
        self.aug_policy_schedule.append(['layer: '+str(self.current_layer_index), best_policy[1], best_policy[2]])
        
        mean_weights = np.empty([self.current_train_data.shape[1]])
        rand_val_acc, rand_test_acc, best_val_acc, best_train_index, best_val_index = [0]*5
        
        layer_index=self.current_layer_index
        
        mean_weights = train_result[7]
        rand_val_acc += train_result[0]
        rand_test_acc += train_result[1]

        self.feature_weights = mean_weights
        val_stack, test_prob, test_stack, layer = train_result[3:-1]

        if layer_index == 1:
            self.current_ensemble_prob = test_prob
        if layer_index > 1:
            self.current_ensemble_prob = np.copy(self.current_ensemble_prob)
            self.current_ensemble_prob += test_prob
            
        ensemble_prob = self.current_ensemble_prob / (layer_index + 1)
        
        if layer_index > 0:
            print('ensemble acc: ',compute_accuracy(self.current_test_label, ensemble_prob ))
        
        if self.num_classes == 2:
            val_stack = val_stack[:, 0::2]
            test_stack = test_stack[:, 0::2]

        self.current_train_data = np.concatenate([self.back_train_data, val_stack],axis=1)
        self.current_test_data = np.concatenate([self.back_test_data, test_stack], axis=1 )

        print("val  acc:{} \nTest acc: {}".format( str(rand_val_acc), str(rand_test_acc)) )
        
        if rand_val_acc >= self.best_val_acc:
            self.best_val_acc = rand_val_acc
            self.best_layer_index = layer_index
            self.best_test_acc = rand_test_acc
            
        self.current_layer_index = self.current_layer_index + 1
        
        self.model.append(layer)
        
        return rand_val_acc, rand_test_acc
    
    def get_best_acc_of_all_layer(self):
        print("best val acc " +str(self.best_val_acc))
        print("best layer index "+str(self.best_layer_index))
        print("best test acc "+str(self.best_test_acc))

    def predict(self, test_data, y_test, ensemble_layer=1):
        test_data_new = test_data.copy()
        
        ensemble_prob = []

        layer_index = 0
        for layer in self.model:
            print('layer ',layer_index)
            test_prob, test_stack = predict_KfoldWarpper(self, layer, test_data_new)
            
            if self.num_classes == 2:
                test_stack = test_stack[:, 0::2]
            
            print('test acc: ',compute_accuracy(y_test, test_prob))

            test_data_new = np.concatenate([test_data, test_stack], axis=1)
            
            if layer_index >= ensemble_layer:
                ensemble_prob.append(test_prob)
                print('ensemble acc: ',compute_accuracy(y_test, np.array(ensemble_prob).reshape(-1,1) / (layer_index + 1 - ensemble_layer)))
            layer_index = layer_index + 1
        
        return np.argmax(test_prob, axis=1)
    
