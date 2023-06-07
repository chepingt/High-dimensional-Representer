import os, sys
import numpy as np 
import scipy.stats as ss
import argparse
import pandas as pd
import seaborn as sns
import scipy.sparse as sp

def get_influence_functions(X_trn, Y_trn, theta, X_tst, Y_tst, C):
    n_train, d = X_trn.shape
    n_test, d = X_tst.shape
    nonzero_indices= np.nonzero(theta)[0]
    
    X_trn_proj = X_trn[:, nonzero_indices]
    X_tst_proj = X_tst[:, nonzero_indices].reshape(n_test, -1)
    theta_proj = theta[nonzero_indices,:].reshape(-1,1)
    
    if isinstance(X_trn,  sp.csr.csr_matrix):
        exp_t = np.exp((X_trn_proj.dot(theta_proj) ).reshape(n_train, 1)) # n_train * 1 
        hessian_scalar =  exp_t / np.multiply(1 + exp_t, 1 + exp_t )
        hessian_scalar = hessian_scalar.reshape(n_train, 1) 
        hes =  X_trn_proj.multiply(hessian_scalar) # n_train * p_nonzero
        H = (X_trn_proj.T @ hes).toarray() + 1e-6 * np.identity(len(nonzero_indices))
        inverse_H = np.linalg.inv(H)
        
        grad_scalar = ( - Y_trn + exp_t / (1 + exp_t) ).reshape(n_train, 1)	 # n_train * 1
        train_grad = X_trn_proj.multiply( grad_scalar ).toarray()
        train_grad += 1 / C * np.sign(theta_proj.reshape(-1))
        test_grad = X_tst_proj.toarray()
        train_grad_H = np.matmul(train_grad, inverse_H)
         
        if_importance = - np.matmul(train_grad_H , test_grad.T)
        
    else:
        exp_t = np.exp( np.matmul(X_trn_proj, theta_proj).reshape(n_train, 1)) # n_train * 1 
        hessian_scalar =  exp_t / np.multiply(1 + exp_t, 1 + exp_t )
        hessian_scalar = hessian_scalar.reshape(n_train, 1) 
        hes = np.multiply( X_trn_proj, hessian_scalar) # n_train * p_nonzero
        
        H = np.matmul(hes.T, X_trn_proj) + 1e-6 * np.identity(len(nonzero_indices))
        inverse_H = np.linalg.inv(H)
        
        grad_scalar = ( - Y_trn + exp_t / (1 + exp_t) ).reshape(-1,1)	 # n_train * 1
        
        train_grad = np.multiply( grad_scalar, X_trn_proj )
        train_grad += 1 / C * np.sign(theta_proj.reshape(-1))
        train_grad_H = np.matmul(train_grad, inverse_H)
        
        test_grad = X_tst_proj
        
        if_importance = - np.matmul(train_grad_H , test_grad.T)
        
    return if_importance

def get_high_dim_representers(X_trn, Y_trn, theta, X_tst, Y_tst, C):
    n_train, d = X_trn.shape
    n_test, d = X_tst.shape
    nonzero_indices = np.nonzero(theta)[0]
    X_trn_proj = X_trn[:, nonzero_indices]
    X_tst_proj = X_tst[:, nonzero_indices]
    theta_proj = theta[nonzero_indices,:].reshape(-1)
    abs_theta_proj = np.absolute(theta_proj)
    if isinstance(X_trn, sp.csr.csr_matrix):
        exp_t = np.exp((X_trn_proj.dot(theta_proj) ).reshape(n_train, 1)) # n_train * 1 
        grad_scalar = ( - Y_trn + exp_t / (1 + exp_t) ).reshape(n_train, 1)	 # n_train * 1
        similarities_high_dim = X_trn_proj @ (X_tst_proj.multiply(abs_theta_proj)).T # n_train * n_test
        importance_high_dim = - similarities_high_dim.multiply(grad_scalar).toarray() # n_train * n_test
    else: 
        exp_t = np.exp( np.matmul(X_trn_proj, theta_proj).reshape(n_train, 1)) # n_train * 1 
        grad_scalar = ( - Y_trn + exp_t / (1 + exp_t) ).reshape(n_train, 1)	 # n_train * 1

        similarities_high_dim = np.matmul(np.multiply( X_trn_proj, abs_theta_proj.reshape(1,-1) ), X_tst_proj.T) # n_train * n_test
        importance_high_dim = - np.multiply(grad_scalar, similarities_high_dim) # n_train * n_test, 
    return importance_high_dim

def get_l2_representers(X_trn, Y_trn, theta, X_tst, Y_tst, C):
    n_train, d = X_trn.shape
    n_test, d = X_tst.shape
    if isinstance(X_trn,  sp.csr.csr_matrix):
        exp_t = np.exp((X_trn.dot(theta) ).reshape(n_train, 1)) # n_train * 1 
        grad_scalar = ( - Y_trn + exp_t / (1 + exp_t) ).reshape(n_train, 1)	 # n_train * 1
        similarities_l2 = X_trn @ X_tst.T # n_train * n_test
        importance_l2 = - similarities_l2.multiply(grad_scalar).toarray() # n_train * n_test
    
    else:
        t = np.matmul(X_trn, theta).reshape(n_train,1) # n_train * 1 
        grad_scalar = ( - Y_trn + np.exp(t) / (1 + np.exp(t)) ).reshape(-1,1)	 # n_train * 1
        
        similarities_l2 = np.matmul(X_trn, X_tst.T) # n_train * n_test
        importance_l2 = - np.multiply(grad_scalar, similarities_l2) # n_train * n_test 
    return importance_l2