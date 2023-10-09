echo "start......"
# we do not implement other aug_type besides 'cutmix'

# search policy on df for adult and kdd


# dataset="adult"
# aug_type="cutmix"
# max_features=0.2
# random_state=42

# str="dataset=${dataset},\n aug_type=${aug_type},\n max_features=${max_features},\n random_state=${random_state}\n"
# pathname="${dataset}_${aug_type}_${max_features}_${random_state}_search.log"
# echo -e $str
# nohup python -u aug.py --dataset ${dataset} --aug_type ${aug_type} --max_features ${max_features}  --random_state ${random_state} >records/${pathname}


# dataset=kdd
# aug_type=cutmix
# max_features=0.3
# random_state=42

# str="dataset=${dataset},\n aug_type=${aug_type},\n max_features=${max_features},\n random_state=${random_state}\n"
# pathname="${dataset}_${aug_type}_${max_features}_${random_state}_search.log"
# echo -e $str
# nohup python -u aug.py --dataset ${dataset} --aug_type ${aug_type} --max_features ${max_features}  --random_state ${random_state} >records/${pathname}

# use policy on df for adult and kdd


dataset="adult"
aug_type="cutmix"
max_features=0.2
random_state=42

str="dataset=${dataset},\n aug_type=${aug_type},\n max_features=${max_features},\n random_state=${random_state}\n"
pathname="${dataset}_${aug_type}_${max_features}_${random_state}_use.log"
echo -e $str
nohup python -u gcforest.py --dataset ${dataset} --aug_type ${aug_type} --max_features ${max_features}  --random_state ${random_state} >records/${pathname}

dataset=kdd
aug_type=cutmix
max_features=0.3
random_state=42

str="dataset=${dataset},\n aug_type=${aug_type},\n max_features=${max_features},\n random_state=${random_state}\n"
pathname="${dataset}_${aug_type}_${max_features}_${random_state}_use.log"
echo -e $str
nohup python -u gcforest.py --dataset ${dataset} --aug_type ${aug_type} --max_features ${max_features}  --random_state ${random_state} >records/${pathname}
