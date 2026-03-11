"""Data utils functions for pre-processing and data loading."""
import os
import pickle as pkl
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

from scipy import sparse
import logging

import pandas as pd

def process_data(args, adj , features, labels):
    ## Load data
    data = {'adj_train': sp.csr_matrix(adj), 'features': features, 'labels': labels}
    adj = data['adj_train']

    ## TAKES a lot of time

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
                adj, args.val_prop, args.test_prop, args.seed
        )

    ## TAKES a lot of time
    data['adj_train'] = adj_train
    data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false
    if args.val_prop + args.test_prop > 0:
        data['val_edges'], data['val_edges_false'] = val_edges, val_edges_false
        data['test_edges'], data['test_edges_false'] = test_edges, test_edges_false
    all_info=""

    ## Adj matrix
    adj = data['adj_train']
    data['adj_train_enc'], data['features'] = process(
            data['adj_train'], data['features'], args.normalize_adj, args.normalize_feats
    )

    if args.lambda_rec:
        data['adj_train_dec'] = rowwise_normalizing(data['adj_train'])

    adj_2hop = get_adj_2hop(adj)
    data['adj_train_enc_2hop'] = symmetric_laplacian_smoothing(adj_2hop)

    # NOTE : Re-adjust labels
    # Some data omit `0` class, thus n_classes are wrong with `max(labels)+1`
    args.n_classes = int(data['labels'].max() + 1)

    data['idx_all'] =  range(data['features'].shape[0])
    data_info = "Dataset {} Loaded : dimensions are adj:{}, edges:{}, features:{}, labels:{}\n".format(
            'ddos2019', data['adj_train'].shape, data['adj_train'].sum(), data['features'].shape, data['labels'].shape)
    data['info'] = data_info
    return data

def process(adj, features, normalize_adj, normalize_feats):
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    return adj, features

def get_adj_2hop(adj):
    adj_self = adj + sp.eye(adj.shape[0])
    adj_2hop = adj_self.dot(adj_self)
    adj_2hop.data = np.clip(adj_2hop.data, 0, 1)
    adj_2hop = adj_2hop - sp.eye(adj.shape[0]) - adj
    return adj_2hop

def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def symmetric_laplacian_smoothing(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = adj + sp.eye(adj.shape[0])  # self-loop

    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def rowwise_normalizing(adj):
    """Row-wise normalize adjacency matrix."""
    adj = adj + sp.eye(adj.shape[0])  # self-loop
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv = np.power(rowsum, -1.0).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    return adj.dot(d_mat_inv).transpose().tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def mask_edges(adj, val_prop, test_prop, seed):
    np.random.seed(seed)  # get tp edges
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    # get tn edges
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()   #  LONG
    neg_edges = np.array(list(zip(x, y)))   #  EVEN LONGER
    np.random.shuffle(neg_edges)  # ALSO LONG

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train, torch.LongTensor(train_edges), torch.LongTensor(train_edges_false), torch.LongTensor(val_edges), \
           torch.LongTensor(val_edges_false), torch.LongTensor(test_edges), torch.LongTensor(
            test_edges_false)  
