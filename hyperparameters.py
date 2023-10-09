eprobs = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 1.0]
emags = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] 
nprobs = [0.0, 0.05, 0.1, 0.2, 0.3]
nmags = [0.05, 0.1, 0.2, 0.3, 0.5]
# we give two example, you can search for other dataset as we guide in the ipynb file
aug_schedule = {
    'adult': [['layer: 0', 0.2, 0.5], ['layer: 1', 0.5, 0.4], ['layer: 2', 0.5, 0.4], ['layer: 3', 0.5, 0.4], ['layer: 4', 0.5, 0.4], ['layer: 5', 0.5, 0.4], ['layer: 6', 0.5, 0.4], ['layer: 7', 0.7, 0.2], ['layer: 8', 0.7, 0.1], ['layer: 9', 0.7, 0.1], ['layer: 10', 0.2, 0.6], ['layer: 11', 0.2, 0.6], ['layer: 12', 0.2, 0.6], ['layer: 13', 0.2, 0.5], ['layer: 14', 0.2, 0.5]],
    'kdd': [['layer: 0', 0.0, 0.1], ['layer: 1', 1.0, 0.5], ['layer: 2', 1.0, 0.5], ['layer: 3', 0.05, 0.1], ['layer: 4', 0.0, 0.5], ['layer: 5', 0.0, 0.3], ['layer: 6', 0.0, 0.3], ['layer: 7', 0.0, 0.3], ['layer: 8', 0.0, 0.3], ['layer: 9', 0.0, 0.3], ['layer: 10', 0.0, 0.3], ['layer: 11', 0.0, 0.3], ['layer: 12', 0.0, 0.3], ['layer: 13', 0.0, 0.3], ['layer: 14', 0.0, 0.3]]
}
