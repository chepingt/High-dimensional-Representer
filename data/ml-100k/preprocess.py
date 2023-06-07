import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np

dataset_dir = 'ml-100k'
test_set_ratio = 0.5
valid_set_ratio = 0
np.random.seed(1)

names = ['user_id', 'item_id', 'rating', 'timestamp']
ratings_df = pd.read_csv(os.path.join(dataset_dir, 'u.data'), sep='\t', names=names)
X = ratings_df[['user_id', 'item_id']].values 
X[:,0] = X[:,0] - 1 # index should start from 0
X[:,1] = X[:,1] - 1 
y = ratings_df['rating'].values

n_users = np.max(X[:,0]) + 1
n_items = np.max(X[:,1]) + 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_ratio + valid_set_ratio)

with open('ml-100k-ex.train.rating', 'w') as f:
    for i in range(len(y_train)):   
        f.write("{} {} {}\n".format(X_train[i][0], X_train[i][1], y_train[i]))

with open('ml-100k-ex.test.rating', 'w') as f:
    for i in range(len(y_test)):   
        f.write("{} {} {}\n".format(X_test[i][0], X_test[i][1], y_test[i]))
