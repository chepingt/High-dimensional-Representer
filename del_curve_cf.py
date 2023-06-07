import sys, os
import numpy as np
import pandas as pd
import random
import argparse
import torch 
import time
import yaml
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import pickle as pkl
from attrdict import AttrDict

from src.dataset import load_dataset
from src.models import get_model 
from src.explainers_for_cf import HighDimRepresenter, FastInfluenceAnalysisMF,\
    FastInfluenceAnalysisPytorch, Random, TracInCP
from src.recorder import Recorder

def run(args, config, algos, n_removes):
    train_ds, valid_ds, test_ds = load_dataset(args.dataset_dir, args.dataset, normalize = not args.unnormalize)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=config.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)
    
    n_users, n_items = train_loader.dataset.get_dims()
    model = get_model(n_users, n_items, config.device, config, not args.unnormalize)
    
    model.fit(train_loader, valid_loader, False, 
              config.save_checkpoint_periods, args.model_dir)
    mse_loss, l1_loss = model.evaluate(test_loader)
    print("MSE Error (Test):", mse_loss)
    print("MAE Error (Test):", l1_loss)
    # Dump to model folder 
    #model.dump_embedding_weights(os.path.dirname(args.model_path))
    
    user_embeddings, item_embeddings = model.get_embedding_weights()
    test_indices = np.random.choice(len(test_ds), args.n_test_per_run)
    
    Algo2Explainer = {
        'high-dim rep': HighDimRepresenter(user_embeddings, item_embeddings, train_ds, 'square_error', normalize = True),
        'FIA': FastInfluenceAnalysisPytorch(train_ds, model, True),
        'TracIn': TracInCP(train_ds, model, args.model_dir),
        'Random': Random(train_ds, args.method)
    }
    
    explainer_list = [ Algo2Explainer[algo] for algo in algos ] 
    
    D = {}
    for algo in algos:
        D[algo] = [ [] for n in n_removes ]
    
    for test_idx in tqdm(test_indices):
        user_id, item_id, label = test_ds[test_idx]
        test_pred = model.get_single_prediction(user_id, item_id)
        
        score_list = []
        indices_list = []
        for algo, explainer in zip(algos, explainer_list):
            item_expl, user_expl = explainer.explain(user_id, item_id)
            scores = np.zeros(len(train_ds))
            
            for train_idx, score in item_expl:
                scores[train_idx] = score
                
            for train_idx, score in user_expl:
                scores[train_idx] = score
            
            if args.method == 'opponent':
                indices = np.argsort(-scores, axis = 0) 
            else:
                indices = np.argsort(scores, axis = 0) 
            score_list.append(scores)
            indices_list.append(indices)
            print(algo)
            for i in range(1,11):
                print(indices[-i], scores[indices[-i]])
            
            for n_id, n  in enumerate(n_removes):
                n_remain = len(train_ds) - n
                res = remove_and_retrain(args, config, indices, n_remain, 
                                         train_ds, user_id, item_id ) 
                pred_difference = res - test_pred 
                D[algo][n_id].append(pred_difference)
    return D

def remove_and_retrain(args, config, indices, n_remain, train_ds, 
                       target_user_id, target_item_id):
    # expls : (item_expl, user_expl)
    remaining_indices = indices[:n_remain] 
    new_train_ds = train_ds.get_dataset_with_indices(remaining_indices)
    
    train_loader = DataLoader(new_train_ds, batch_size=config.batch_size, shuffle=True)
    
    n_users, n_items = train_loader.dataset.get_dims()
    model = get_model(n_users, n_items, config.device, config, not args.unnormalize)
    model.fit(train_loader)
    
    pred_score = model.get_single_prediction(target_user_id, target_item_id)
    
    return pred_score

def test_model_hyperparameter(args, config):
    # used to tune model hyperparameter
    train_ds, valid_ds, test_ds = load_dataset(args.dataset_dir, args.dataset, not args.unnormalize)
    train_loader = DataLoader(train_ds, batch_size = config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size = config.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size = config.batch_size, shuffle=False)
    
    n_users, n_items = train_loader.dataset.get_dims()
    model = get_model(n_users, n_items, config.device, config, not args.unnormalize)
    
    model.fit(train_loader, valid_loader, True)
    mse_loss, l1_loss = model.evaluate(test_loader)
    print("MSE Error (Test):", mse_loss)
    print("MAE Error (Test):", l1_loss)
    print()
    
def parse_arguments():
    """Parse training arguments"""
    parser = argparse.ArgumentParser() 
    
    parser.add_argument( "--dataset", type = str, 
                        choices = ['ml-100k', 'ml-1m', 'yelp', 'amazon_art', 'amazon_video_games'],
                        default = 'ml-100k')
    parser.add_argument( "--config", type = str, default = 'config/YoutubeDNN-ml100k.yaml')
    parser.add_argument( "--dataset_dir", type = str, default = 'data')
    parser.add_argument( "--model_dir", type = str, default = None)
    parser.add_argument( "--exp_dir", type = str, default = None)
    parser.add_argument( "--seed", type = int, default = 1)
    parser.add_argument( "--test_model", action = 'store_true')
    
    parser.add_argument( "--n_runs", type = int, default = 30)
    parser.add_argument( "--n_test_per_run", '-n_test', type = int, default = 10)
    parser.add_argument( "--gpu", type = int, default = -1)
    parser.add_argument( "--unnormalize", action = 'store_true') # normalize the ratings from [1,5] to [-1,1]
    parser.add_argument( "--method", "-m", type = str, 
                        choices = ['positive', 'negative'], # drop positive or negative impact samples
                        default = 'positive')
    return parser

def main():
    parser = parse_arguments()
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
 
    with open(args.config, 'r') as f: 
        config = yaml.safe_load(f)
        config = AttrDict(config)
    
    if torch.cuda.is_available() and args.gpu >= 0:  
        config.device = "cuda:{}".format(args.gpu)
    else:  
        config.device = "cpu"
    
    if args.exp_dir is None:
        args.exp_dir =  'exp/delcurve/{}/{}/runs{}_n_test_{}_{}_{}'.format(args.dataset, config.model,
                            args.n_runs, args.n_test_per_run, args.method, suffix)
    
    if args.model_dir is None:
        args.model_dir = os.path.join(args.exp_dir, 'checkpoints')
    print(args)
    print(config)
     
    if args.test_model:
        test_model_hyperparameter(args, config) 
        exit()
    
    os.makedirs(args.exp_dir, exist_ok = True)
    os.makedirs(args.model_dir, exist_ok = True)
    
    if config.model == 'SoftImputeMF':
        algos = ['high-dim rep', 'Random']
    else:
        algos = ['high-dim rep', 'TracIn' ,'FIA', 'Random']
            
    if args.dataset == 'ml-100k':
        n_removes = [ 5, 10, 15, 20, 25 ]  
    elif args.dataset == 'ml-1m': 
        n_removes = [ 10, 20, 30, 40 ,50 ]  
    elif args.dataset == 'yelp': 
        n_removes = [ 3, 6, 9, 12, 15 ]  
    elif args.dataset == 'amazon_art': 
        n_removes = [ 3, 6, 9, 12, 15 ]  
    elif args.dataset == 'amazon_video_games': 
        n_removes = [ 3, 6, 9, 12, 15 ]  
    
    print("n_removes:", n_removes)
    recorder = Recorder(n_removes, algos) 
    
    for i in range(args.n_runs):
        print("Run {}:".format(i+1))
        D = run(args, config, algos, n_removes)
        recorder.update(D) 
        recorder.print() 
        log_path = os.path.join(args.exp_dir, 'run{}.log'.format(i)) 
        recorder.print(log_path) 
    
        dump_path = os.path.join(args.exp_dir, 'recorder{}.pkl'.format(i))
        pkl.dump(recorder, open( dump_path, 'wb'))

if __name__ == '__main__':
    main()