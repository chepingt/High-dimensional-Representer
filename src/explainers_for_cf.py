import sys, os
import numpy as np
from sklearn.utils.extmath import randomized_svd
import time
from torch.utils.data import DataLoader
from torch import nn
import torch 
import copy

class Random():
    def __init__(self, train_ds, method):
        self.n_train = len(train_ds)
        self.user2item = train_ds.user2item
        self.item2user = train_ds.item2user 
        self.method = method
    def explain(self, user_id, item_id):
        
        item_based_data = [] 
        related_items, item_related_train_id = self.user2item[user_id]
        
        user_based_data = []
        related_users, user_related_train_id = self.item2user[item_id]
         
        if self.method == 'negative':
            if len(related_items) > 0:
                rand_scores = -np.random.rand(len(item_related_train_id)) 
                item_based_data = [ ( item_related_train_id[i], rand_scores[i]) for i in range(len(item_related_train_id))]
            if len(related_users) > 0:
                rand_scores = -np.random.rand(len(user_related_train_id))
                user_based_data = [ ( user_related_train_id[i], rand_scores[i]) for i in range(len(user_related_train_id))]
        else:
            if len(related_items) > 0:
                rand_scores = np.random.rand(len(item_related_train_id))
                item_based_data = [ ( item_related_train_id[i], rand_scores[i]) for i in range(len(item_related_train_id))]
            if len(related_users) > 0:
                rand_scores = np.random.rand(len(user_related_train_id))
                user_based_data = [ ( user_related_train_id[i], rand_scores[i]) for i in range(len(user_related_train_id))]

        return item_based_data, user_based_data
    
class HighDimRepresenter():
    def __init__(self, user_embeddings, item_embeddings, train_ds, loss_type, 
                 negative_sample_weight = 1, normalize = True):
        # train_Xs: (user_i, item_i), i = 1,...,n 
        # train_Ys: R^n
        # d_loss: take two inputs: x_i, y_i
        self.n_users, self.embedding_dim = user_embeddings.shape
        self.n_items, _ = item_embeddings.shape
        self.n_train = len(train_ds)
        self.user2item = train_ds.user2item
        self.item2user = train_ds.item2user
        self.negative_sample_weight = negative_sample_weight 
        # Compute normalized embeddings
        start_time = time.time()
        
        if normalize:
            U1, Sigma1, V1 = randomized_svd(user_embeddings, n_components=self.embedding_dim, n_iter=20, random_state=10)
            U2, Sigma2, V2 = randomized_svd(item_embeddings.T, n_components=self.embedding_dim, n_iter=20, random_state=10)
            
            mid = np.matmul(V1, U2) 
            mid = np.multiply(Sigma1, mid.T).T
            mid = np.multiply(mid, Sigma2)
            
            U3, Sigma3, V3 = randomized_svd(mid, n_components=self.embedding_dim, n_iter=20, random_state=10)
            
            U = np.matmul(U1, U3)
            V = np.matmul(V3, V2)
            Sigma = Sigma3
            
            self.normalized_user_embedding = np.multiply(U, np.sqrt(Sigma))
            self.normalized_item_embedding = np.multiply(V.T, np.sqrt(Sigma))
        else:
            self.normalized_user_embedding = user_embeddings
            self.normalized_item_embedding = item_embeddings
             
        # TODO: use torch.auto.grad to make it applicable to any loss 
        if loss_type == 'square_error':
            d_loss = self.__square_loss_derivative
        elif loss_type == 'bce':
            d_loss = self.__BCE_derivative 
    
        inner_product = np.sum(np.multiply(user_embeddings[train_ds.user_list], item_embeddings[train_ds.item_list] ), 1)
        
        self.global_importance = - d_loss(inner_product, train_ds.y)
        
        print("Preprocessing time (s):", time.time() - start_time)

    def explain(self, user_id, item_id):
        
        item_based_data = [] 
        related_items, item_related_train_id = self.user2item[user_id]
        if len(related_items) > 0:
            item_local_sim = np.matmul(self.normalized_item_embedding[related_items], 
                                self.normalized_item_embedding[item_id].reshape(-1,1))
            item_global_importance = self.global_importance[item_related_train_id]
            item_importance = np.multiply(item_local_sim, item_global_importance)
            item_based_data = list(zip(item_related_train_id, item_importance))
         
        user_based_data = []
        related_users, user_related_train_id = self.item2user[item_id]
        if len(related_users) > 0:
            user_local_sim = np.matmul(self.normalized_user_embedding[related_users], 
                                self.normalized_user_embedding[user_id].reshape(-1,1))
            user_global_importance = self.global_importance[user_related_train_id] 
            user_importance = np.multiply(user_local_sim, user_global_importance)
            user_based_data = list(zip(user_related_train_id, user_importance))
            
        return item_based_data, user_based_data

    def __square_loss_derivative(self, Y_pred, Y):
        return (Y_pred - Y).reshape(-1, 1)
    
    def __BCE_derivative(self, Y_pred, Y):
        Y = Y.reshape(-1, 1)
        Y_pred = Y_pred.reshape(-1,1)
        grad = ( - Y + np.exp(Y_pred) / (1 + np.exp(Y_pred)) ).reshape(-1,1)
        # rewighting
        grad = np.multiply(grad, (1 - Y) * self.negative_sample_weight + Y * 1  )
        return grad
        

class FastInfluenceAnalysisMF():
    def __init__(self, user_embeddings, item_embeddings, train_ds):
        # current implementation is only for square loss
        self.n_users, self.embedding_dim = user_embeddings.shape
        self.n_items, _ = item_embeddings.shape
        self.train_ds = train_ds
        
        self.n_train = len(train_ds)
        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings
        
        self.user2item = self.train_ds.user2item
        self.item2user = self.train_ds.item2user 
        self.X = train_ds.X
        self.Y = train_ds.y
        
        inner_product = np.sum(np.multiply(user_embeddings[train_ds.user_list], item_embeddings[train_ds.item_list] ), 1)
        self.train_user_grads = np.multiply((inner_product - self.Y), self.item_embeddings[self.train_ds.item_list].T ).T 
        self.train_item_grads = np.multiply((inner_product - self.Y), self.user_embeddings[self.train_ds.user_list].T ).T 
            
    def get_inverse_hvp(self, user_id, item_id, expl_type = 'user', damping_factor = 1e-6):
        if expl_type == 'item':
            related_items, item_related_train_id = self.user2item[user_id]
            H = np.matmul(self.item_embeddings[related_items].reshape(-1, self.embedding_dim, 1),
                          self.item_embeddings[related_items].reshape(-1 , 1, self.embedding_dim)) 
            H = np.sum(H, axis = 0) + damping_factor * np.identity(self.embedding_dim)
            inverse_H = np.linalg.inv(H)
            #print(inverse_H) 
            return np.matmul(inverse_H, self.item_embeddings[item_id].reshape(self.embedding_dim, 1))
        elif expl_type == 'user':
            related_users, user_related_train_id = self.item2user[item_id]
            H = np.matmul(self.user_embeddings[related_users].reshape(-1, self.embedding_dim, 1),
                          self.user_embeddings[related_users].reshape(-1 , 1, self.embedding_dim)) 
            H = np.sum(H, axis = 0) + damping_factor * np.identity(self.embedding_dim)
            inverse_H = np.linalg.inv(H)
            
            return np.matmul(inverse_H, self.user_embeddings[user_id].reshape(self.embedding_dim, 1))
    
    def explain(self, user_id, item_id):
        item_based_data = [] 
        related_items, item_related_train_id = self.user2item[user_id]
        if len(related_items) > 0:
            inv_hvp = self.get_inverse_hvp(user_id, item_id, 'item')
            item_importance = -np.matmul(self.train_user_grads[item_related_train_id], inv_hvp  )
            item_based_data = list(zip(item_related_train_id, item_importance))
        
        user_based_data = []
        related_users, user_related_train_id = self.item2user[item_id]
        if len(related_users) > 0: 
            inv_hvp = self.get_inverse_hvp(user_id, item_id, 'user')
            user_importance = -np.matmul(self.train_item_grads[user_related_train_id], inv_hvp )
            user_based_data = list(zip(user_related_train_id, user_importance))
        print(item_based_data[:5])
        print(user_based_data[:5])
        
        return item_based_data, user_based_data

class FastInfluenceAnalysisPytorch():
    def __init__(self, train_ds, model, record_inverse_hessian = False):
        # current implementation is only for square loss
        self.train_ds = train_ds
        self.model = model
        self.loss_func = copy.deepcopy(model.loss_func)
        self.n_train = len(train_ds)
        self.record_inverse_hessian = record_inverse_hessian
        self.inverse_hessians_item = {}
        self.inverse_hessians_user = {}
        
        self.user2item = self.train_ds.user2item
        self.item2user = self.train_ds.item2user 
        self.X = train_ds.X
        self.Y = train_ds.y

    def get_inverse_hessian_and_grads(self, related_train_id, expl_type = 'user', damping_factor = 1e-6):
        model = self.model
        if expl_type == 'user':
            user_id = self.train_ds.user_list[related_train_id[0]]
            if self.record_inverse_hessian and user_id in self.inverse_hessians_user:
                inverse_H, grads = self.inverse_hessians_user[user_id]
            else:
                u_emb = model.lookup_embedding(int(user_id), 'user')   
                item_ids = self.train_ds.item_list[related_train_id] 
                i_embs = model.lookup_embedding(item_ids, 'item') 
                
                scores = self.train_ds.y[related_train_id]
                scores = torch.FloatTensor(scores)
                
                all_one_matrix = torch.ones([len(related_train_id), 1]).to(model.device)
                def f(u_emb):
                    u_embs = torch.matmul(all_one_matrix, u_emb.reshape(1, -1))
                    batch_pred = model.forward_from_embeddings(u_embs, i_embs)
                    self.loss_func.reduction = 'mean'
                    loss = self.loss_func(batch_pred, scores.float().to(self.model.device) ) * len(related_train_id) 
                    return loss
                
                hessian = torch.autograd.functional.hessian(f, u_emb)
                H = hessian.cpu().detach().numpy().reshape(model.embedding_dim, model.embedding_dim)
                H += damping_factor *  np.identity(model.embedding_dim)
                inverse_H = np.linalg.inv(H)
                
                def f2(u_emb):
                    u_embs = torch.matmul(all_one_matrix, u_emb.reshape(1, -1))
                    batch_pred = model.forward_from_embeddings(u_embs, i_embs)
                    self.loss_func.reduction = 'none'
                    loss = self.loss_func(batch_pred, scores.float().to(self.model.device) ) 
                    return loss
                grads = torch.autograd.functional.jacobian(f2, u_emb)
                grads = grads.cpu().detach().numpy().reshape(len(related_train_id), model.embedding_dim)
                
                self.inverse_hessians_user[user_id] = (inverse_H, grads)
            
        elif expl_type == 'item':
            item_id = self.train_ds.item_list[related_train_id[0]]
            if self.record_inverse_hessian and item_id in self.inverse_hessians_item:
                inverse_H, grads  = self.inverse_hessians_item[item_id]    
            else:
                i_emb = model.lookup_embedding(int(item_id), 'item')   
                user_ids = self.train_ds.user_list[related_train_id] 
                u_embs = model.lookup_embedding(user_ids, 'user') 
                
                scores = self.train_ds.y[related_train_id]
                scores = torch.FloatTensor(scores)
                
                all_one_matrix = torch.ones([len(related_train_id), 1]).to(model.device)
                def f(i_emb):
                    i_embs = torch.matmul(all_one_matrix, i_emb.reshape(1, -1))
                    batch_pred = model.forward_from_embeddings(u_embs, i_embs)
                    self.loss_func.reduction = 'mean'
                    loss = self.loss_func(batch_pred, scores.float().to(self.model.device) ) * len(related_train_id)
                    return loss
                
                hessian = torch.autograd.functional.hessian(f, i_emb)
                H = hessian.cpu().detach().numpy().reshape(model.embedding_dim, model.embedding_dim)
                H += damping_factor *  np.identity(model.embedding_dim)
                inverse_H = np.linalg.inv(H)
                
                def f2(i_emb):
                    i_embs = torch.matmul(all_one_matrix, i_emb.reshape(1, -1))
                    batch_pred = model.forward_from_embeddings(u_embs, i_embs)
                    self.loss_func.reduction = 'none'
                    loss = self.loss_func(batch_pred, scores.float().to(self.model.device) )
                    return loss
                grads = torch.autograd.functional.jacobian(f2, i_emb)
                grads = grads.cpu().detach().numpy().reshape(len(related_train_id), model.embedding_dim)
                
                self.inverse_hessians_item[item_id] = (inverse_H, grads)
        
        return inverse_H, grads 
    
    def get_pred_grad(self, user_id, item_id, expl_type = 'user'):
        model = self.model
        u_emb = model.lookup_embedding(int(user_id), 'user')   
        i_emb = model.lookup_embedding(int(item_id), 'item')   
        if expl_type == 'user':
            def f(x):
                pred = model.forward_from_embeddings(x, i_emb)
                return pred
            grad = torch.autograd.functional.jacobian(f, u_emb)
        elif expl_type == 'item':
            def f(x):
                pred = model.forward_from_embeddings(u_emb, x)
                return pred
            grad = torch.autograd.functional.jacobian(f, i_emb)
        grad = grad.cpu().detach().numpy().reshape( model.embedding_dim, 1)
        return grad
    
    def explain(self, user_id, item_id, damping_factor = 1e-6):
        item_based_data = [] 
        related_items, item_related_train_id = self.user2item[user_id]
        if len(related_items) > 0:
            inverse_H, train_grads = self.get_inverse_hessian_and_grads(item_related_train_id, 'user')
            pred_grad = self.get_pred_grad(user_id, item_id, 'user')
            item_importance = - np.matmul(train_grads, np.matmul(inverse_H, pred_grad ) )
            item_based_data = list(zip(item_related_train_id, item_importance))
        
        user_based_data = []
        related_users, user_related_train_id = self.item2user[item_id]
        if len(related_users) > 0:
            inverse_H, train_grads = self.get_inverse_hessian_and_grads(user_related_train_id, 'item')
            pred_grad = self.get_pred_grad(user_id, item_id, 'item')
            user_importance = - np.matmul(train_grads, np.matmul(inverse_H, pred_grad ) )
            user_based_data = list(zip(user_related_train_id, user_importance))
        
        return item_based_data, user_based_data

class TracInCP():
    def __init__(self, train_ds, model, model_cp_dir):
        # current implementation is only for square loss
        self.train_ds = train_ds
        self.n_train = len(train_ds)
        self.model = model
        self.model_cp_dir = model_cp_dir
        self.loss_func = copy.deepcopy(model.loss_func)
        
        self.user2item = self.train_ds.user2item
        self.item2user = self.train_ds.item2user 
        self.X = train_ds.X
        self.Y = train_ds.y
    
    def get_pred_grad(self, model, user_id, item_id):
        model.eval()
        test_pred = model.forward(torch.LongTensor(np.array([user_id])).to(model.device), 
                       torch.LongTensor(np.array([item_id])).to(model.device))
        test_pred.backward()
        grad = torch.cat([param.grad.reshape(-1) for param in model.parameters()])
        model.zero_grad()
        return grad.cpu().detach().numpy()
    
    def get_loss_grad(self, model, user_id, item_id, label):
        model.eval()
        label = torch.FloatTensor(np.array([label])).to(model.device) 
        test_pred = model.forward(torch.LongTensor(np.array([user_id])).to(model.device), 
                       torch.LongTensor(np.array([item_id])).to(model.device))
        loss = self.loss_func(test_pred, label)
        loss.backward()
        grad = torch.cat([param.grad.reshape(-1) for param in model.parameters()])
        model.zero_grad()
        
        return grad.cpu().detach().numpy()
    
    def explain(self, user_id, item_id):
        model = self.model
        related_items, item_related_train_id = self.user2item[user_id]
        item_based_data = [[train_id,0] for train_id in item_related_train_id] 
        
        related_users, user_related_train_id = self.item2user[item_id]
        user_based_data = [[train_id,0] for train_id in user_related_train_id] 
         
        for file_path in  os.listdir(self.model_cp_dir ):
            if not file_path.endswith('.ckpt'):
                continue
            model_path = os.path.join(self.model_cp_dir, file_path)
            model.load(model_path, model.device)
            model.eval()
            test_grad = self.get_pred_grad(model, user_id, item_id)
            
            for j, train_item_id in enumerate(related_items):
                label = self.train_ds.y[item_related_train_id[j] ]
                pred_grad = self.get_loss_grad(model, user_id, train_item_id, label )
                item_importance = - np.dot(test_grad, pred_grad )
                item_based_data[j][1] += item_importance
            
            for j, train_user_id in enumerate(related_users):
                label = self.train_ds.y[user_related_train_id[j] ]
                pred_grad = self.get_loss_grad(model, train_user_id, item_id, label)
                user_importance = - np.dot(test_grad, pred_grad )
                user_based_data[j][1] += user_importance
        self.model.load_last_checkpoint(self.model_cp_dir)   
        
        return item_based_data, user_based_data