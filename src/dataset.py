from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np

def load_dataset(dataset_dir, dataset_name, normalize = True, binarized = False, 
                 flipping_ratio = 0.1, negative_sampling_ratio = 10):
    dataset_list = []
    n_users, n_items = 0, 0
    for name in ['train', 'valid', 'test']:
        file_path = os.path.join(dataset_dir, '{}/{}-ex.{}.rating'.format(dataset_name, 
                                                                          dataset_name, name)) 
        user_ids, item_ids, scores = [], [], []
            
        with open(file_path, 'r') as f:
            for line in f:
                user_id, item_id, score = line.split()
                user_ids += [int(user_id)]
                item_ids += [int(item_id)]
                if '.' in score:
                    scores += [int(float(score))]
                else:
                    scores += [int(score)]
        n_users = max(np.max(user_ids) + 1, n_users)
        n_items = max(np.max(item_ids) + 1, n_items)
        X = np.array(list(zip(user_ids, item_ids)))
        scores = np.array(scores)
        
        preprocess = True if name == 'train' else False
        nsr = negative_sampling_ratio if name == 'train' else 0
        fr = flipping_ratio if name == 'train' else 0
        
        if binarized:
            dataset_list += [BinarizedRecommenderDataset(X, scores, n_users, n_items, fr, nsr, preprocess = True) ]
        else:
            dataset_list += [RecommenderDataset(X, scores, n_users, n_items, normalize, preprocess) ]
    
    for ds in dataset_list:
        ds.n_users = n_users
        ds.n_items = n_items
        
    return dataset_list

def load_dataset_for_dataset_debbugging(dataset_dir, dataset_name):
    dataset_list = []
    n_users, n_items = 0, 0
    X_list, score_list = [], [] 
    for name in ['train', 'test']:
        file_path = os.path.join(dataset_dir, '{}/{}-ex.{}.rating'.format(dataset_name, 
                                                                          dataset_name, name)) 
        user_ids, item_ids, scores = [], [], []
            
        with open(file_path, 'r') as f:
            for line in f:
                user_id, item_id, score = line.split()
                user_ids += [int(user_id)]
                item_ids += [int(item_id)]
                scores += [int(score)]
        n_users = max(np.max(user_ids) + 1, n_users)
        n_items = max(np.max(item_ids) + 1, n_items)
        X = np.array(list(zip(user_ids, item_ids)))
        scores = np.array(scores)
        X, scores = binarized(X, scores)
        
        X_list.append(X)
        score_list.append(scores)

    train_dataset = BinarizedRecommenderDataset2(X_list[0], n_users, n_items, X_list[1], True)
    test_dataset = BinarizedRecommenderDataset2(X_list[1], n_users, n_items, is_train = False) 
    
    X_train, X_valid, y_train, y_valid= train_test_split(X_list[0], score_list[0], test_size=0.1)
    parital_train_dataset =  BinarizedRecommenderDataset2(X_train, n_users, n_items, X_list[1], True)
    valid_dataset =  BinarizedRecommenderDataset2(X_valid, n_users, n_items, is_train = False) 
        
    return [train_dataset, test_dataset, parital_train_dataset, valid_dataset]

def load_dataset_for_dataset_debbugging_v2(dataset_dir, dataset_name):
    dataset_list = []
    n_users, n_items = 0, 0
    X_list, score_list = [], [] 
    for name in ['train', 'test']:
        file_path = os.path.join(dataset_dir, '{}/{}-ex.{}.rating'.format(dataset_name, 
                                                                          dataset_name, name)) 
        user_ids, item_ids, scores = [], [], []
            
        with open(file_path, 'r') as f:
            for line in f:
                user_id, item_id, score = line.split()
                user_ids += [int(user_id)]
                item_ids += [int(item_id)]
                scores += [int(score)]
        n_users = max(np.max(user_ids) + 1, n_users)
        n_items = max(np.max(item_ids) + 1, n_items)
        X = np.array(list(zip(user_ids, item_ids)))
        scores = np.array(scores)
        X, scores = binarized(X, scores)
        
        X_list.append(X)
        score_list.append(scores)

    train_dataset = BinarizedRecommenderDataset2(X_list[0], n_users, n_items, X_list[1], True)
    test_dataset = BinarizedRecommenderDataset2(X_list[1], n_users, n_items, is_train = False) 
    
    #X_train, X_valid, y_train, y_valid= train_test_split(X_list[0], score_list[0], test_size=0.1)
    #parital_train_dataset =  BinarizedRecommenderDataset2(X_train, n_users, n_items, X_list[1], True)
    valid_dataset =  BinarizedRecommenderDataset2(X_list[0], n_users, n_items, is_train = False) 
        
    return [train_dataset, valid_dataset, test_dataset]

class RecommenderDataset(Dataset):
    def __init__(self, X, y, n_users, n_items, normalize, preprocess = False):
        self.X = X 
        self.y = y
        self.n_users = n_users
        self.n_items = n_items
        self.normalize = normalize
        
        if normalize:
            self.y = (self.y - 3) / 2 
        
        if preprocess:
            self.preprocess()
        
        # users
        self.user_list = self.X[:,0]
        self.item_list = self.X[:,1] 
    
    def get_dims(self):
        return self.n_users, self.n_items 
     
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx][0], self.X[idx][1], self.y[idx]
        
    def preprocess(self):
        # get dictionaries
        self.user2item = {i:{'user_id':[], 'train_id':[]} for i in range(self.n_users)} 
        self.item2user = {i:{'item_id':[], 'train_id':[]}  for i in range(self.n_items)}
        
        for train_id, (user_id, item_id) in enumerate(self.X):
            self.user2item[user_id]['user_id'] += [item_id]
            self.user2item[user_id]['train_id'] += [train_id]
            self.item2user[item_id]['item_id'] += [user_id]
            self.item2user[item_id]['train_id'] += [train_id]
        
        for i in range(self.n_users):
            self.user2item[i] = (np.array(self.user2item[i]['user_id']), np.array(self.user2item[i]['train_id']))
            
        for i in range(self.n_items):
            self.item2user[i] = (np.array(self.item2user[i]['item_id']), np.array(self.item2user[i]['train_id']) )
    
    def get_dataset_with_indices(self, remaining_indices):
        X = self.X[remaining_indices]
        y = self.y[remaining_indices]
        
        return RecommenderDataset(X, y, self.n_users, self.n_items, False)

class BinarizedRecommenderDataset(RecommenderDataset):
    def __init__(self, X, y, n_users, n_items, flipping_ratio = 0.1,
                 negative_sampling_ratio = 20, preprocess = False):
        super(BinarizedRecommenderDataset, self).__init__(X, y, n_users, n_items, False,
                                                          preprocess = False)
        self.flipping_ratio = flipping_ratio
        self.negative_sampling_ratio = negative_sampling_ratio
		
        # Do negative sampling before fitting models
        X_pos, y_pos = self.binarized(X, y) 
        
        print("Number of positive samples:", len(X_pos))
        if negative_sampling_ratio > 0:
            X_neg, y_neg = self.negative_sampling(X_pos, y_pos)
            print("Number of negative samples:", len(X_neg))
        
        self.flipping_indices = np.array([-1])
        if flipping_ratio > 0:
            X_pos, y_pos, flipping_indices = self.flip(X_pos, y_pos, flipping_ratio)
            self.flipping_indices = flipping_indices
            print("Number of flipped samples:", len(self.flipping_indices))
        
        if negative_sampling_ratio > 0:
            self.X = np.vstack((X_pos, X_neg))
            self.y = np.concatenate((y_pos, y_neg))
        else:
            self.X = X_pos
            self.y = y_pos
        
        self.user_list = self.X[:,0]
        self.item_list = self.X[:,1] 
        
        if preprocess:
            self.preprocess()
   
    def binarized(self, X, y):    
        remaining_indices = np.nonzero(y > 3)[0]
        X_pos = X[remaining_indices]
        y_pos = y[remaining_indices]
        y_pos[y_pos > 3] = 1
        
        return X_pos, y_pos
    
    def flip(self, X_pos, y_pos, flipping_ratio):
        n_flipped = int(len(X_pos) * flipping_ratio)
        flipping_indices = np.random.choice(len(X_pos), n_flipped, replace = False)
        
        new_X = np.array(X_pos)
        new_y = np.array(y_pos)
        new_y[flipping_indices] = 0
        
        return new_X, new_y, flipping_indices
    
    def negative_sampling(self, X, y):
        # y should be binarized
            
        n_negatives = int(len(X) * self.negative_sampling_ratio)
        counted = set()
        
        for x, y in X:
            counted.add((x, y))
        
        if self.negative_sampling_ratio >= 20:
            # sample the entire dataset
            n_negatives = self.n_users * self.n_items - len(X) 
            X_neg = np.zeros((n_negatives, 2))
            y_neg = np.zeros(n_negatives)
            cnt = 0
            for i in range(self.n_users):
                for j in range(self.n_items):
                    if (i,j) not in counted:
                        X_neg[cnt, 0] = i
                        X_neg[cnt, 1] = j
                        cnt += 1
        else: 
            N = 2 * n_negatives    
            users = np.random.choice(self.n_users, N )
            items = np.random.choice(self.n_items, N )
            cnt = 0
            now = 0
            X_neg = np.zeros((n_negatives, 2))
            y_neg = np.zeros(n_negatives)
            while cnt < n_negatives:
                now_user, now_item = users[now], items[now]
                if (now_user, now_item) not in counted:
                    X_neg[cnt, 0] = now_user
                    X_neg[cnt, 1] = now_item
                    counted.add((now_user, now_item)) 
                    cnt += 1
                now += 1
                if now >= N:
                    users = np.random.choice(self.n_users, N )
                    items = np.random.choice(self.n_items, N )
                    now = 0
                if len(counted) > self.n_users * self.n_items - 10:
                    break 

        return X_neg.astype(int), y_neg.astype(int) 
    
def binarized(X, y):    
    remaining_indices = np.nonzero(y > 3)[0]
    X_pos = X[remaining_indices]
    y_pos = y[remaining_indices]
    y_pos[y_pos > 3] = 1
    
    return X_pos, y_pos

class BinarizedRecommenderDataset2(RecommenderDataset):
    def __init__(self, X, n_users, n_items, X_fn = None, is_train = False):
        #super(BinarizedRecommenderDataset2, self).__init__(X, y, n_users, n_items, False,
        #                                                  preprocess = False)
        self.n_users = n_users
        self.n_items = n_items
        X_pos = X 
        y_pos = np.ones(len(X))
        print("Number of positive samples:", len(X_pos))
        if is_train:
            X_neg, y_neg, flipping_indices = self.negative_sampling(X_pos, X_fn)
        
            self.flipping_indices = flipping_indices + len(X_pos)
        
            self.X = np.vstack((X_pos, X_neg))
            self.y = np.concatenate((y_pos, y_neg))
        else:
            self.X = X_pos
            self.y = y_pos
             
        print("Number of all samples:", len(self.X))
        
        self.user_list = self.X[:,0]
        self.item_list = self.X[:,1] 
    
        self.preprocess()
    
    def negative_sampling(self, X, X_fn):
        # y should be binarized
            
        counted = set()
        neg_set = set() 
        for x, y in X:
            counted.add((x, y))
            
        for x, y in X_fn:
            neg_set.add((x, y))
        
        flipping_indices = []
        # sample the entire dataset
        n_negatives = self.n_users * self.n_items - len(X) 
        X_neg = np.zeros((n_negatives, 2))
        y_neg = np.zeros(n_negatives)
        cnt = 0
        for i in range(self.n_users):
            for j in range(self.n_items):
                if (i,j) not in counted:
                    X_neg[cnt, 0] = i
                    X_neg[cnt, 1] = j
                    if (i,j) in neg_set:
                        flipping_indices.append(cnt)
                    cnt += 1

        return X_neg.astype(int), y_neg.astype(int), np.array(flipping_indices)