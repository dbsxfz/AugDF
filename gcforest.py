from layer import *
import ray
from data.dataset import get_data
from sklearn.preprocessing import OneHotEncoder
from hyperparameters import *
import argparse
class gcForest:
    def __init__(self, classifier, num_estimator, encoder, num_forests, num_classes, means, std, max_layer=15, 
            max_features=0.1, n_fold=5, min_samples_leaf=1, sample_weight=None, gpu_id=0, random_state=42, weights=[None],
                purity_function="gini" , bootstrap=True, parallel=True, num_threads=-1, extend=1, aug_type='erase', aug_prob=eprobs, aug_mag = emags,aug_policy_schedule=[],ray=False):
        self.gpu_id = gpu_id
        self.classifier = classifier
        self.encoder = encoder
        self.num_estimator = num_estimator
        self.num_forests = num_forests
        self.num_classes = num_classes
        self.n_fold = n_fold
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
        self.aug_prob_list = aug_prob
        self.aug_mag_list = aug_mag

        self.model = []
        self.current_layer_index = 0
        self.current_ensemble_prob=0
        self.best_layer_index = 0
        self.best_val_acc = 0
        self.best_test_acc = 0
        
        self.kf_N = n_fold
        self.kf_val = StratifiedKFold( self.kf_N, shuffle=True, random_state=random_state)#default as 42
        self.aug_policy_schedule = aug_policy_schedule
        self.search_schedule=[]
        self.ray=ray
                      
    def load_data(self, train_data, train_label):
        
        num_classes = int(np.max(train_label) + 1)
        if( num_classes != self.num_classes ):
            print("num_classes is"+str(num_classes)+", while given num_classes is "+str(self.num_classes)+".")
            raise Exception("init num_classes not equal to actual num_classes")
        
        self.current_train_data = train_data
        self.current_train_label = train_label
            
        self.back_train_data = train_data
        self.back_train_label = train_label
        
        self.num_features = train_data.shape[1]
    # try will not add a new layer to the model, which is used to find a best policy  
    @ray.remote
    def train_KfoldWarpper_ray(self, train_data, train_label, prob, mag, kf):
        return train_KfoldWarpper(self, train_data, train_label, prob, mag, kf)
              
    def get_best_policy(self,train_data=None, train_label=None):
        if train_data is None:
            train_data=self.current_train_data
        if train_label is None:
            train_label=self.current_train_label
        layer_index = self.current_layer_index
        kf = StratifiedKFold( self.n_fold, shuffle=True, random_state = self.random_state + self.current_layer_index)
        print("\n--------------\nlayer {},   X_train shape:{}...\n ".format(str(layer_index), train_data.shape) )

        best_rand_val_acc = 0.0
        best_rand_policy = []
        X_train = train_data
        y_train = train_label          
        temp = []  
        result_set = [(j, k) for j in self.aug_prob_list for k in self.aug_mag_list if k >= j + 0.1]
        if self.ray==True:
            for (j,k) in result_set:
                temp.append(self.train_KfoldWarpper_ray.remote(self, X_train, y_train,prob=j, mag=k, kf=kf))
            temp_result = ray.get(temp) 
        elif self.ray==False:
            for (j,k) in result_set:
                temp.append(train_KfoldWarpper(self, X_train, y_train,prob=j, mag=k, kf=kf))
            temp_result=temp  
        for i, temp in enumerate(temp_result):   
            rand_val_acc = compute_accuracy(train_label, temp[0])      
            j = result_set[i][0]
            k = result_set[i][1]
            print(self.aug_type,' ',j,' ',k,' ', rand_val_acc,' ')
            if rand_val_acc >= best_rand_val_acc:
                best_rand_val_acc = rand_val_acc
                best_rand_policy = (self.aug_type, j, k)
        
        print('final best policy, ', best_rand_policy)
        return best_rand_policy       
    
    def fit(self,train_data, train_label):
        self.current_train_data = train_data
        self.current_train_label = train_label
        self.num_classes = int(np.max(train_label) + 1)
        self.back_train_data = train_data
        self.back_train_label = train_label
        self.num_features = train_data.shape[1]

        if len(self.aug_policy_schedule)==self.max_layer:
            for i in range(self.max_layer):
                self.fit_one_layer(self.aug_policy_schedule[i])
        else:
            print("no enough aug_policy_schedule for max_layer")
            
    def fit_one_layer(self,best_policy):
        layer_index=self.current_layer_index
        kf = StratifiedKFold( self.n_fold, shuffle=True, random_state=self.random_state + self.current_layer_index)
        val_prob, val_stack, layer, mean_weights = train_KfoldWarpper(self, train_data=self.current_train_data, train_label=self.current_train_label, prob=best_policy[1], mag=best_policy[2], kf=kf)
        temp_val_acc = compute_accuracy(self.current_train_label, val_prob)
        self.feature_weights = mean_weights
        if self.num_classes == 2:
            val_stack = val_stack[:, 0::2]
        self.current_train_data = np.concatenate([self.back_train_data, val_stack],axis=1)
        print("layer index:{}".format( str(layer_index)) )
        print("val  acc:{} ".format( str(temp_val_acc)) )       
        if temp_val_acc >= self.best_val_acc:
            self.best_val_acc = temp_val_acc
            self.best_layer_index = layer_index        
        self.current_layer_index = self.current_layer_index + 1
        self.model.append(layer)
        self.search_schedule.append(best_policy)
        return temp_val_acc           
    
    def get_best_acc_of_all_layer(self):
        print("best val acc " +str(self.best_val_acc))
        print("best layer index "+str(self.best_layer_index))
             
    # more detail type comparing to real score function
    def score(self, test_data, y_test, ensemble_layer=1):
        
        test_data_new = test_data.copy()
        ensemble_prob = 0

        layer_index = 0
        for layer in self.model:
            print('layer ',layer_index)
            test_prob, test_stack = predict_KfoldWarpper(self, layer, test_data_new)
            
            if self.num_classes == 2:
                test_stack = test_stack[:, 0::2]
            
            print('test acc: ',compute_accuracy(y_test, test_prob))

            test_data_new = np.concatenate([test_data, test_stack], axis=1)
            if layer_index == ensemble_layer:
                self.current_ensemble_prob = test_prob
            if layer_index > ensemble_layer:
                self.current_ensemble_prob = np.copy(self.current_ensemble_prob)
                self.current_ensemble_prob += test_prob
                ensemble_prob = self.current_ensemble_prob / (layer_index + 1)# need renew
                print('ensemble acc: ',compute_accuracy(y_test, ensemble_prob ))
            layer_index = layer_index + 1
        
        return np.argmax(ensemble_prob, axis=1)
    
    def predict_proba(self,test_data,ensemble_layer=1):
        test_data_new = test_data.copy()
        layer_index = 0
        for layer in self.model:
            test_prob, test_stack = predict_KfoldWarpper(self, layer, test_data_new)
            
            if self.num_classes == 2:
                test_stack = test_stack[:, 0::2]
            test_data_new = np.concatenate([test_data, test_stack], axis=1)
            if layer_index == ensemble_layer:
                self.current_ensemble_prob = test_prob
            if layer_index > ensemble_layer:
                self.current_ensemble_prob = np.copy(self.current_ensemble_prob)
                self.current_ensemble_prob += test_prob
            
            ensemble_prob = self.current_ensemble_prob / (layer_index + 1)
            layer_index = layer_index + 1
        return ensemble_prob
    
    def predict(self,test_data,ensemble_layer=1):
        ensemble_prob=self.predict_proba(test_data,ensemble_layer)
        return np.argmax(ensemble_prob, axis=1)

    def get_search_schedule(self):
        return self.search_schedule
    
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
    one_hot = OneHotEncoder(sparse_output=False).fit(y_train.reshape(-1,1))
    clf = GradientBoosting('bce', lr=0.3, colsample=0.3, verbose=-1)
    clf.fit(X_train, one_hot.transform(y_train.reshape(-1,1)))
    weights = clf.get_feature_importance()
    weights = [w / sum(weights) for w in weights]
    aug_policy_schedule = aug_schedule[args.dataset]
    df = gcForest(encoder=one_hot, weights=weights, classifier=args.classifier,num_classes=num_classes,\
        max_layer=args.max_layer, gpu_id = args.gpu_id, aug_type=args.aug_type,\
        ray=args.use_ray,max_features=args.max_features,random_state=args.random_state,\
        means=np.mean(X_train, axis=0),std=np.std(X_train, axis=0),aug_policy_schedule=aug_policy_schedule,\
        num_forests=args.num_forests,num_estimator=args.num_estimator)
    df.fit(X_train, y_train)
    df.score(X_test,y_test)
    if args.use_ray:
        ray.shutdown()