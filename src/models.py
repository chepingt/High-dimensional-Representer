import os, sys
import torch
from torch import nn
import numpy as np
from collections import OrderedDict
from sklearn.utils.extmath import randomized_svd
from src.soft_impute import SoftImpute

def get_model(n_users, n_items, device, config, dataset_normalized):
    model_name = config.model
    if model_name == 'MF':
        return MF( n_users, n_items, device, config, dataset_normalized)
    elif model_name == 'YoutubeDNN':
        return YoutubeDNN( n_users, n_items, device, config, dataset_normalized)
    elif model_name == 'SoftImputeMF':
        return SoftImputeMF( n_users, n_items, config)

class BaseModel(nn.Module):
    def __init__(self, n_users, n_items, device, config, dataset_normalized):
        super(BaseModel, self).__init__()
        self.n_users = n_users 
        self.n_items = n_items
        self.embedding_dim = config.embedding_dim
        self.device = device
        self.config = config
        self.dataset_normalized = dataset_normalized
        
        if config.loss == 'mse':
            self.loss_func = nn.MSELoss()
        elif config.loss == 'bce':
            weight = torch.FloatTensor([1 / config.negative_sample_weight])
            self.loss_func = nn.BCEWithLogitsLoss(pos_weight = weight)
        
        self.user_embeddings = nn.Embedding(n_users, self.embedding_dim)
        self.item_embeddings = nn.Embedding(n_items, self.embedding_dim)
        
        if dataset_normalized: 
            self.user_embeddings.weight.data.uniform_(-1 / self.embedding_dim, 1 / self.embedding_dim)
            self.item_embeddings.weight.data.uniform_(-1 / self.embedding_dim, 1 / self.embedding_dim)
        else:
            self.user_embeddings.weight.data.uniform_(0, 10 / self.embedding_dim)
            self.item_embeddings.weight.data.uniform_(0, 10 / self.embedding_dim)
    
    def save(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)
    
    def load(self, checkpoint, device):
        self.load_state_dict(torch.load(checkpoint, map_location=device))
        self.to(device)
        self.device = device
    
    def fit(self, train_loader, valid_loader = None, verbose = False,
                 save_checkpoint_periods = 0, save_model_dir = None):
        self.to(self.device)
        if self.config.optimizer is None or self.config.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate, 
                                     betas=(0.9, 0.999), eps=1e-08, weight_decay=self.config.weight_decay)
        elif self.config.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.config.learning_rate, momentum=0.9) 
        for epoch_id in range(1, self.config.epochs+1):
            self.train()
            total_loss = []
            for idx, (user_ids, item_ids, scores) in enumerate(train_loader):
                batch_pred = self.forward(user_ids.to(self.device), item_ids.to(self.device))
                
                loss = self.loss_func(batch_pred.reshape(-1,1), scores.reshape(-1,1).float().to(self.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += [loss.item()]
            
            if verbose and epoch_id % 5 ==0 and valid_loader is not None:
                print("Epoch {}".format(epoch_id))
                print("Training loss of epoch {}:{}".format(epoch_id,np.mean(total_loss)))
                if self.config.loss == 'bce': 
                    recall = self.evaluate_recall(valid_loader, 20) 
                    print("Recall@20 (Valid)", recall)
                else:
                    mse_loss, l1_loss = self.evaluate(valid_loader)
                    print("MSE Error (Valid):", mse_loss)
                    print("MAE Error (Valid):", l1_loss)
            
            if save_model_dir is not None and epoch_id % save_checkpoint_periods == 0:
                save_path = os.path.join(save_model_dir, '{}.ckpt'.format(epoch_id))
                self.save(save_path)
    
    def load_last_checkpoint(self, save_model_dir):
        save_path = os.path.join(save_model_dir, '{}.ckpt'.format(self.config.epochs))
        self.load(save_path, self.device)
         
    def get_training_loss(self, train_loader):
        self.loss_func.reduction = 'none'
        losses = []
        for idx, (user_ids, item_ids, scores) in enumerate(train_loader):
            batch_pred = self.forward(user_ids.to(self.device), item_ids.to(self.device))
            
            loss = self.loss_func(batch_pred.reshape(-1,1), scores.reshape(-1,1).float().to(self.device))
            losses.append(loss.cpu().detach().numpy())
        losses = np.vstack(losses)
        self.loss_func.reduction = 'mean'
        return losses 
    
    def evaluate(self, loader):
        mse_loss_fn = nn.MSELoss()
        l1_loss_fn = nn.L1Loss()
        self.eval()
        
        total_loss = [[],[]]
        for idx, (user_ids, item_ids, scores) in enumerate(loader):
            batch_pred = self.forward(user_ids.to(self.device), item_ids.to(self.device))
            
            mse_loss = mse_loss_fn(batch_pred, scores.to(self.device))
            l1_loss  = l1_loss_fn(batch_pred, scores.to(self.device))
            
            total_loss[0] += [mse_loss.item()]
            total_loss[1] += [l1_loss.item()]

        total_loss[0] = np.mean(total_loss[0])
        total_loss[1] = np.mean(total_loss[1])
        
        return total_loss

    def evaluate_recall(self, loader, k = 20):
        user_embeddings, item_embeddings = self.get_embedding_weights()
        scores = np.matmul(user_embeddings, item_embeddings.T)
        
        arg_indices = np.argsort(-scores, axis = 1)
        
        ds = loader.dataset 
        recalls = []
        for i in range(ds.n_users):
            true_items = ds.user2item[i][0]
            topk_items = arg_indices[i][:k]
            hit_items = set(list(true_items)) & set(list(topk_items))
            recall = len(hit_items) / (len(true_items) + 1e-12)
            recalls.append(recall)
        
        return np.mean(recalls)
    
    def get_single_prediction(self, user_id, item_id):
        test_pred = self.forward(torch.LongTensor(np.array([user_id])).to(self.device), 
                       torch.LongTensor(np.array([item_id])).to(self.device)).item()
        return test_pred
    
    def lookup_embedding(self, ids, lookup_type = 'user'):
        if isinstance(ids, int):
            ids = np.array([ids])
        elif isinstance(ids, list):
            ids = np.array(ids)
        ids = torch.LongTensor(ids).to(self.device)
        if lookup_type == 'user':
            return self.user_embeddings(ids)
        elif lookup_type == 'item':
            return self.item_embeddings(ids)
        else:
            raise NameError(lookup_type)

     
class MF(BaseModel):
    def __init__(self, n_users, n_items, device, config, dataset_normalized):
        super(MF, self).__init__(n_users, n_items, device, config, dataset_normalized)
        
    def forward(self, user_ids, item_ids):
        u_emb = self.user_embeddings(user_ids)
        i_emb = self.item_embeddings(item_ids)
        outputs = torch.sum(torch.multiply(u_emb, i_emb), 1).reshape(-1)
        return outputs
    
    def forward_from_embeddings(self, user_embeddings, item_embeddings):
        i_emb = item_embeddings
        u_emb = user_embeddings
        outputs = torch.sum(torch.multiply(u_emb, i_emb), 1).reshape(-1)
        
        return outputs
    
    def get_embedding_weights(self):
        return self.user_embeddings.weight.cpu().detach().numpy(), \
            self.item_embeddings.weight.cpu().detach().numpy()
    
class YoutubeDNN(BaseModel):
    def __init__(self, n_users, n_items, device, config, dataset_normalized):
        super(YoutubeDNN, self).__init__(n_users, n_items, device, config, dataset_normalized)
        hidden_units = [self.config.embedding_dim] + list(self.config.user_hidden_units)
        user_mlp_layers = []
        
        for idx in range(len(hidden_units) - 1):
            user_mlp_layers.append(nn.Linear(hidden_units[idx], hidden_units[idx + 1]))
            if idx < len(hidden_units) -2:
                #user_mlp_layers.append(nn.BatchNorm1d(hidden_units[idx+1]))
                user_mlp_layers.append(nn.ReLU()) 
                user_mlp_layers.append(nn.Dropout(p = self.config.dropout_p)) 
        self.user_tower = nn.Sequential(*user_mlp_layers)

    def forward_from_embeddings(self, user_embeddings, item_embeddings):
        i_emb = item_embeddings
        u_emb = self.user_tower(user_embeddings)
        outputs = torch.sum(torch.multiply(u_emb, i_emb), 1).reshape(-1)
        
        return outputs
        
    def forward(self, user_ids, item_ids):
        u_emb = self.user_embeddings(user_ids)
        u_emb = self.user_tower(u_emb)
        i_emb = self.item_embeddings(item_ids)
        outputs = torch.sum(torch.multiply(u_emb, i_emb), 1).reshape(-1)
        
        return outputs
    
    def get_embedding_weights(self):
        now = 0
        user_embs = []
        while now < self.n_users:
            start = now
            end = now + self.config.batch_size
            if end > self.n_users:
                end = self.n_users
            batch_idx = torch.LongTensor(np.arange(start,end)).to(self.device)
            embs = self.user_embeddings(batch_idx) 
            embs = self.user_tower(embs)
            user_embs.append(embs.cpu().detach().numpy())
            now = end
        user_embs = np.vstack(user_embs)
        return user_embs, self.item_embeddings.weight.cpu().detach().numpy()

class SoftImputeMF():
    def __init__(self, n_users, n_items, config):
        # can only be applied to normalized datasets, i.e. the average of the labels should be around 0
        self.n_users = n_users 
        self.n_items = n_items
        self.config = config
        self.loss_func =  nn.MSELoss()
        
        self.embedding_dim = config.embedding_dim 
        self.user_embeddings = np.zeros((n_users, self.embedding_dim))
        self.item_embeddings = np.zeros((n_items, self.embedding_dim)) 
    
    def fit(self, train_loader, valid_loader = None, verbose = False,
            save_checkpoint_periods = 0, save_model_dir = None):
        train_ds = train_loader.dataset
        
        X_obv = np.zeros((self.n_users, self.n_items)) - 100
        X_obv[train_ds.user_list, train_ds.item_list ] = train_ds.y 
        observed_mask = np.zeros((self.n_users, self.n_items)) 
        observed_mask = np.where(X_obv > -99, True, False)
        
        X_pred, lambda_reg, pred_rank = SoftImpute(verbose = verbose, max_iters = self.config.max_iters, 
                                               min_value = -1, max_value = 1, max_rank = self.embedding_dim).solve(X_obv, ~observed_mask)

        U, Sigma, V = randomized_svd(X_pred, n_components = pred_rank, n_iter=20, random_state=10)
        
        self.user_embedding = np.multiply(U, np.sqrt(Sigma))
        self.item_embedding = np.multiply(V.T, np.sqrt(Sigma))
        
        if valid_loader and verbose:
            mse, mae = self.evaluate(valid_loader)
            print("MSE Error (Valid):", mse)
            print("MAE Error (Valid):", mae)
             
    def evaluate(self, loader):
        dataset = loader.dataset
        label = dataset.y
        
        inner_product = np.sum(np.multiply(self.user_embeddings[dataset.user_list], self.item_embeddings[dataset.item_list] ), 1)
        inner_product[inner_product > 1] = 1 
        inner_product[inner_product < -1] = -1 
        mse_loss = np.mean((label - inner_product) ** 2)
        mae_loss = np.mean(np.abs(label - inner_product))
        
        return mse_loss, mae_loss
    
    def get_embedding_weights(self):
        return self.user_embedding, self.item_embedding
    
    def save(self, checkpoint):
        self.user_embedding.dump(open(checkpoint + '.user', 'wb'))
        self.item_embedding.dump(open(checkpoint + '.item', 'wb'))
    
    def load(self, checkpoint, device):
        self.user_embedding = np.load(open(checkpoint + '.user', 'rb'))
        self.item_embedding = np.load(open(checkpoint + '.item', 'rb'))
        self.to(device)

    def get_single_prediction(self, user_id, item_id):
        return np.sum(np.multiply(self.user_embedding[user_id], self.item_embedding[item_id] )) 