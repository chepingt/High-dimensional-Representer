import os, sys
import numpy as np 
import scipy.stats as st
import scipy.sparse as sp
from tqdm import tqdm
import argparse
import pandas as pd
import seaborn as sns
import time
import pickle as pkl
from multiprocessing import Pool
from functools import partial
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import src.istarmap as istarmap # import to apply patch
from src.recorder import Recorder
from src.explainers_for_l1reg import get_influence_functions, get_high_dim_representers, get_l2_representers 

def remove_and_retrain(X_trn, Y_trn, C, indices, n_remain, X_tst, Y_tst):
    indices = indices[:n_remain]
    n_dim = X_trn.shape[1]
    
    clf = LogisticRegression(penalty = 'l1', fit_intercept = False, C = C, solver ='liblinear').fit(X_trn[indices], Y_trn[indices].reshape(-1))
    theta = clf.coef_.reshape(n_dim, 1)
    if isinstance(X_tst,  sp.csr.csr_matrix):
        pred = (X_tst.dot(sp.csc_matrix(theta))).toarray().reshape(-1) 
    else:
        pred = np.matmul(X_tst, theta).reshape(-1) # n_test * 1
    return pred

def load_dataset(data_dir, dataset_name, seed = -1):
    def read_sparse_data(path, dim):
        labels = []
        row = []
        col = []
        data = []
        with open(path, 'r') as f:
            for data_id, line in enumerate(f):
                tokens = line.split(' ')
                label = int(tokens[0])
                for token in tokens[1:]:
                    if ':' in token:
                        idx, value = token.split(':')
                        idx = int(idx)
                        value = float(value)
                        row.append(data_id)
                        col.append(idx)
                        data.append(value)
                labels.append(label)
        X = sp.csr_matrix((data, (row, col)), shape=(data_id+1, dim))
        y = np.array(labels).reshape(-1,1)
        return X, y
    if dataset_name in ['rcv1', 'gisette'] :
        if dataset_name == 'rcv1':
            train_data_path = os.path.join(data_dir, dataset_name, 'rcv1_train.binary')
            test_data_path = os.path.join(data_dir, dataset_name, 'rcv1_test.binary')
            dim = 47237
        elif dataset_name == 'gisette':
            train_data_path = os.path.join(data_dir, dataset_name, 'gisette_scale')
            test_data_path = os.path.join(data_dir, dataset_name, 'gisette_scale.t')
            dim = 5001
        X_train, y_train = read_sparse_data(train_data_path, dim)
        X_test, y_test = read_sparse_data(test_data_path, dim) 

        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2,random_state= seed)
    elif dataset_name in ['news20']:
        train_data_path = os.path.join(data_dir, dataset_name, 'news20.binary')
        dim = 1355192
        
        X_train, y_train = read_sparse_data(train_data_path, dim)
        
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state= seed)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.11,random_state= seed)
    print(X_train.shape, X_test.shape) 
    print("Loading dataset is done.") 
        
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def compare_run_time(args):
    def get_conf_interval(L):
        a, b = st.norm.interval(alpha=0.95, loc=np.mean(L), scale=st.sem(L))
        return (b-a)/2
         
    X_trn, Y_trn, X_vld, Y_vld, X_tst, Y_tst = load_dataset('data', args.dataset, args.seed)
    C_list = get_C_list(args.dataset)
    
    for C in C_list: 
        clf = LogisticRegression(penalty = 'l1', fit_intercept = False, C = C, solver ='liblinear').fit(X_trn, Y_trn.reshape(-1))
        theta = clf.coef_.reshape(-1, 1)
        high_dim_times, l2_times, if_times = [], [], []
        
        for i in tqdm(range(args.n_test)):
            X_tst_temp = X_tst[i,:].reshape(1,-1)
            Y_tst_temp = Y_tst[i,:].reshape(1,1)
            start_time = time.time()
            train_high_dim_importance = get_high_dim_representers(X_trn, Y_trn, theta, X_tst_temp, Y_tst_temp, C)
            high_dim_time = time.time() - start_time
            high_dim_times.append(high_dim_time)
            
            start_time = time.time()
            train_l2_importance = get_l2_representers(X_trn, Y_trn, theta, X_tst_temp, Y_tst_temp, C)
            l2_time = time.time() - start_time
            l2_times.append(l2_time)
            
            start_time = time.time()
            train_if_importance = get_influence_functions(X_trn, Y_trn, theta, X_tst_temp, Y_tst_temp, C) 
            IF_time = time.time() - start_time
            if_times.append(IF_time)
        print("-"*100)
        print("C=", C)
        print("Number of nonzero entries:", len(np.nonzero(theta)[0]))
        print("High-dim rep:", np.mean(high_dim_times), get_conf_interval(high_dim_times))
        print("L2 rep:", np.mean(l2_times), get_conf_interval(l2_times))
        print("IF:", np.mean(if_times), get_conf_interval(if_times))
    
def run(args, X_trn, Y_trn, X_tst, Y_tst, algos, ps):
    print("C=",args.C)
    clf = LogisticRegression(penalty = 'l1', fit_intercept = False, C = args.C, solver ='liblinear').fit(X_trn, Y_trn.reshape(-1))
    print("Test set Accuracy:{:.2%}".format(clf.score(X_tst, Y_tst.reshape(-1))))
    theta = clf.coef_.reshape(-1, 1)
    print("Number of nonzero entries:", len(np.nonzero(theta)[0]))
    
    indices = np.random.choice(len(Y_tst), args.n_test)
    X_tst_temp = X_tst[indices]
    Y_tst_temp = Y_tst[indices]
    n_train = X_trn.shape[0]
    
    start_time = time.time()
    train_high_dim_importance = get_high_dim_representers(X_trn, Y_trn, theta, X_tst_temp, Y_tst_temp, args.C)
    print(train_high_dim_importance.shape)
    high_dim_time = time.time() - start_time
    print("Time for computing high-dim representers", high_dim_time)
    
    train_l2_importance = get_l2_representers(X_trn, Y_trn, theta, X_tst_temp, Y_tst_temp, args.C)
    
    start_time = time.time()
    train_if_importance = get_influence_functions(X_trn, Y_trn, theta, X_tst_temp, Y_tst_temp, args.C) 
    IF_time = time.time() - start_time
    print("Time for computing Influence functions", IF_time)
    
    train_importance_random = np.random.rand(n_train, args.n_test) 
    if isinstance(X_tst,  sp.csr.csr_matrix):
        original_test_preds = (X_tst.dot(sp.csc_matrix(theta))).toarray().reshape(-1) 
    else:
        original_test_preds = np.matmul(X_tst, theta).reshape(-1)

    chunk_size = 1
    pool = Pool(min(20, args.n_test))
    scores = [train_high_dim_importance, train_l2_importance, train_if_importance, train_importance_random]
    
    if args.method == 'negative':
        indices = [ np.argsort(-score, axis = 0) for score in scores ]
    else:
        indices = [ np.argsort(score, axis = 0) for score in scores ]
    
    n_test = X_tst_temp.shape[0]
    
    F = partial(remove_and_retrain, X_trn, Y_trn, args.C )
    D = {}
    for algo in algos:
        D[algo] = [ [] for p in ps ]
    for indice, algo in zip(indices, algos):
        for p_id, p in enumerate(ps):
            #print(algo, p)
            n_remain = int(n_train * (1 - p))
            F_args = [ (indice[:,i], n_remain, X_tst_temp[i,:], Y_tst_temp[i] ) for i in range(n_test)]
        
            for ind, res in enumerate(pool.istarmap(F, F_args, chunk_size)):
                loss_difference = float(res - original_test_preds[ind])
                D[algo][p_id].append(loss_difference)
    return D

def parse_arguments():
    """Parse training arguments"""
    parser = argparse.ArgumentParser() 
    parser.add_argument( "--exp_dir", type = str, default = None)
    parser.add_argument( "--dataset", type = str, 
                        choices = ['gisette', 'news20', 'rcv1'],
                        default = 'news20')
    parser.add_argument( "--seed", type = int, default = 2023)
    parser.add_argument( "--n_test", type = int, default = 40)
    parser.add_argument( "--n_runs", type = int, default = 40)
    parser.add_argument( "--C", type = float, default = 1)
    parser.add_argument( "--compare_runtime", action = 'store_true')
    parser.add_argument( "--method", "-m", type = str, 
                        choices = ['positive', 'negative'], # drop proponent or opponent
                        default = 'positive',)
    return parser

def main(): 
    parser = parse_arguments()
    args = parser.parse_args()
    print(args) 
 
    np.random.seed(args.seed)   
    if args.compare_runtime:
        compare_run_time(args)
        exit()
        
    if args.exp_dir is None:
        args.exp_dir =  'exp/l1_delcurve/{}/runs{}_n_test_{}_{}'.format(args.dataset,
                            args.n_runs, args.n_test, args.method)
    
    os.makedirs(args.exp_dir, exist_ok = True)
    
    algos = ['high-dim rep.', r'$\ell_2 rep.$', 'IF' ,'Random']
    ps = [0.01 * i for i in range(1,6)] 
    
    X_trn, Y_trn, X_vld, Y_vld, X_tst, Y_tst = load_dataset('data', args.dataset, args.seed)
    
    recorder = Recorder(ps, algos)
    for i in range(args.n_runs):
        print("Run", i)
        D = run(args, X_trn, Y_trn, X_tst, Y_tst, algos, ps)
        recorder.update(D)
        recorder.print()
        log_path = os.path.join(args.exp_dir, 'recorder_C{}_{}.log'.format(args.C, i))
        recorder.print(log_path)
        
        dump_path = os.path.join(args.exp_dir, 'recorder_C{}_{}.pkl'.format(args.C, i))
        pkl.dump(recorder, open( dump_path, 'wb'))

    
if __name__ == '__main__':
    main()