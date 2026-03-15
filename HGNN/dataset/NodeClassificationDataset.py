import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import save_npz, load_npz
from scipy.sparse.linalg import eigsh
import sys
from torch.utils.data import Dataset, DataLoader
from HGNN.utils import *
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool_)

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1)).astype(float)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

class NodeClassificationDataset(Dataset):
    """
    Extend the Dataset class for graph datasets
    """
    def __init__(self, args, logger,adj,features,labels):
        self.args = args
        self.process_data(adj,features,labels)

    def _filling_adjacency_numpy(self,data, N, source_ip_index, destination_ip_index):
        try:
            adjacency = np.zeros((N,N), dtype=bool)
        except Exception as e:
            print(f"An error occurred: {e}")
            
        source_ips = data[:, source_ip_index]
        destination_ips = data[:, destination_ip_index]
        mask = ((source_ips[:, np.newaxis] == source_ips) | (source_ips[:, np.newaxis] == destination_ips) | (destination_ips[:, np.newaxis] == source_ips) | (destination_ips[:, np.newaxis] == destination_ips))
        adjacency[mask] = True
        adjacency = adjacency - np.eye(N)
        return adjacency
    
    def compact_adjacency(self,adj):
        max_neighbors = int(np.max(np.sum(adj, axis=1)))
        shape = (adj.shape[0],max_neighbors)
        c_adj = np.zeros(shape)
        c_adj[:,:] = -1
        indices , neighbors = np.where(adj == 1)

        j=-1
        l = indices[0]
        for i,k in zip(indices,neighbors):
            if i == l:
                j+=1
            else:
                l=i
                j=0
            c_adj[i,j]=int(k)
        return c_adj
    
    def compact_weight_matrix(self,c_adj):
        return np.where(c_adj >= 0, 1, 0)
    
    def one_hot_labels(self,y):
        array  = np.zeros((len(y),2))
        for i,j in zip(range(len(y)),y):
            if j:
                array[i,1]=1
            else:
                array[i,0]=1

        return array
    
    def split_data(self,labels, test_prop,val_prop):
        np.random.seed(self.args.seed)
        #nb_nodes = labels.shape[0]
        #all_idx = np.arange(nb_nodes)
        # pos_idx = labels.nonzero()[0]
        # neg_idx = (1. - labels).nonzero()[0]
        pos_idx = labels[:,1].nonzero()[0]
        neg_idx = labels[:,0].nonzero()[0]
        np.random.shuffle(pos_idx)
        np.random.shuffle(neg_idx)
        pos_idx = pos_idx.tolist()
        neg_idx = neg_idx.tolist()
        nb_pos_neg = min(len(pos_idx), len(neg_idx))
        nb_val = round(val_prop * nb_pos_neg)
        nb_test = round(test_prop * nb_pos_neg)
        idx_val_pos, idx_test_pos, idx_train_pos = pos_idx[:nb_val], pos_idx[nb_val:nb_val + nb_test], pos_idx[
                                                                                                    nb_val + nb_test:]
        idx_val_neg, idx_test_neg, idx_train_neg = neg_idx[:nb_val], neg_idx[nb_val:nb_val + nb_test], neg_idx[
                                                                                                    nb_val + nb_test:]
        return idx_val_pos + idx_val_neg, idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg
    
    def process_data(self, adj,features,labels):
            
        adj = self.compact_adjacency(adj)
        weight = self.compact_weight_matrix(adj)
        adj[adj == -1] = 0

        labels = self.one_hot_labels(labels)

        idx_test, idx_train, idx_val = self.split_data(labels,self.args.test_prop,self.args.val_prop)

        train_mask = sample_mask(idx_train, labels.shape[0])
        val_mask = sample_mask(idx_val, labels.shape[0])
        test_mask = sample_mask(idx_test, labels.shape[0])

        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)
        y_train[train_mask, :] = labels[train_mask, :]
        y_val[val_mask, :] = labels[val_mask, :]
        y_test[test_mask, :] = labels[test_mask, :]

        self.adj = adj
        self.weight = weight

        self.features = preprocess_features(features) if self.args.normalize_feats else features
        self.features = features
        assert np.isnan(features).any()== False
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.train_mask = train_mask.astype(int)
        self.val_mask = val_mask.astype(int)
        self.test_mask = test_mask.astype(int)
        self.args.node_num = self.features.shape[0]
        self.args.input_dim = self.features.shape[1]
        self.args.num_class = y_train.shape[1]
    

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return  {
                  'adj': self.adj,
                  'weight': self.weight,
                  'features': self.features,
                  'y_train' : self.y_train,
                  'y_val' : self.y_val,
                  'y_test' : self.y_test,
                  'train_mask' : self.train_mask,
                  'val_mask' : self.val_mask,
                  'test_mask' : self.test_mask,
                }
