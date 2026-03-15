import sys
import math
import time
import os
import shutil
import torch
import torch.distributions as dist
from torch.autograd import Variable, Function, grad
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import argparse
import torch.nn as nn
import scipy.sparse as sp


def lexpand(A, *dimensions):
    """Expand tensor, adding new dimensions on left."""
    return A.expand(tuple(dimensions) + A.shape)


def rexpand(A, *dimensions):
    """Expand tensor, adding new dimensions on right."""
    return A.view(A.shape + (1,)*len(dimensions)).expand(A.shape + tuple(dimensions))


def assert_no_nan(name, g):
    if torch.isnan(g).any(): raise Exception('nans in {}'.format(name))


def assert_no_grad_nan(name, x):
    if x.requires_grad: x.register_hook(lambda g: assert_no_nan(name, g))


# Classes
class Constants(object):
    eta = 1e-5
    log2 = math.log(2)
    logpi = math.log(math.pi)
    log2pi = math.log(2 * math.pi)
    logceilc = 88                # largest cuda v s.t. exp(v) < inf
    logfloorc = -104             # smallest cuda v s.t. exp(v) > 0
    invsqrt2pi = 1. / math.sqrt(2 * math.pi)
    sqrthalfpi = math.sqrt(math.pi/2)


def logsinh(x):
    # torch.log(sinh(x))
    return x + torch.log(1 - torch.exp(-2 * x)) - Constants.log2


def logcosh(x):
    # torch.log(cosh(x))
    return x + torch.log(1 + torch.exp(-2 * x)) - Constants.log2


class Arccosh(Function):
    # https://github.com/facebookresearch/poincare-embeddings/blob/master/model.py
    @staticmethod
    def forward(ctx, x):
        ctx.z = torch.sqrt(x * x - 1)
        return torch.log(x + ctx.z)

    @staticmethod
    def backward(ctx, g):
        z = torch.clamp(ctx.z, min=Constants.eta)
        z = g / z
        return z


class Arcsinh(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.z = torch.sqrt(x * x + 1)
        return torch.log(x + ctx.z)

    @staticmethod
    def backward(ctx, g):
        z = torch.clamp(ctx.z, min=Constants.eta)
        z = g / z
        return z


# https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.begin = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.begin
        self.elapsedH = time.gmtime(self.elapsed)
        print('====> [{}] Time: {:7.3f}s or {}'
              .format(self.name,
                      self.elapsed,
                      time.strftime("%H:%M:%S", self.elapsedH)))


# Functions
def save_vars(vs, filepath):
    """
    Saves variables to the given filepath in a safe manner.
    """
    if os.path.exists(filepath):
        shutil.copyfile(filepath, '{}.old'.format(filepath))
    torch.save(vs, filepath)


def save_model(model, filepath):
    """
    To load a saved model, simply use
    `model.load_state_dict(torch.load('path-to-saved-model'))`.
    """
    save_vars(model.state_dict(), filepath)


def log_mean_exp(value, dim=0, keepdim=False):
    return log_sum_exp(value, dim, keepdim) - math.log(value.size(dim))


def log_sum_exp(value, dim=0, keepdim=False):
    m, _ = torch.max(value, dim=dim, keepdim=True)
    value0 = value - m
    if keepdim is False:
        m = m.squeeze(dim)
    return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))


def log_sum_exp_signs(value, signs, dim=0, keepdim=False):
    m, _ = torch.max(value, dim=dim, keepdim=True)
    value0 = value - m
    if keepdim is False:
        m = m.squeeze(dim)
    return m + torch.log(torch.sum(signs * torch.exp(value0), dim=dim, keepdim=keepdim))


def get_mean_param(params):
    """Return the parameter used to show reconstructions or generations.
    For example, the mean for Normal, or probs for Bernoulli.
    For Bernoulli, skip first parameter, as that's (scalar) temperature
    """
    if params[0].dim() == 0:
        return params[1]
    # elif len(params) == 3:
    #     return params[1]
    else:
        return params[0]


def probe_infnan(v, name, extras={}):
    nps = torch.isnan(v)
    s = nps.sum().item()
    if s > 0:
        print('>>> {} >>>'.format(name))
        print(name, s)
        print(v[nps])
        for k, val in extras.items():
            print(k, val, val.sum().item())
        quit()


def has_analytic_kl(type_p, type_q):
    return (type_p, type_q) in torch.distributions.kl._KL_REGISTRY


def split_data(labels, test_prop,val_prop):
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    pos_idx = labels.nonzero()[0]
    neg_idx = (1. - labels).nonzero()[0]
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
    
    return idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg, idx_val_pos + idx_val_neg,

def process_data(args, adj,features,labels):
    data = process_data_nc(args,adj,features,labels)
    data['adj_train'], data['features'] = process(
            data['adj_train'], data['features'],args.normalize_adj,args.normalize_feats
    )
    return data

def process_data_nc(args,adj,features,labels):
    idx_test, idx_train , idx_val= split_data(labels, args.test_prop,args.val_prop)
    labels = torch.LongTensor(labels)
    data = {'adj_train': sp.csr_matrix(adj), 'features': features, 'labels': labels, 'idx_train': idx_train,  'idx_test': idx_test , 'idx_val':idx_val}
    return data

def process(adj, features, normalize_adj, normalize_feats):
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats: 
        features = normalize(features)
    features = torch.Tensor(features)
    if normalize_adj:
        adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def create_args(*args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=args[0])
    parser.add_argument('--hidden_dim', type=int, default=args[1])
    parser.add_argument('--num_layers', type=int, default=args[2])
    parser.add_argument('--c', type=int, default=args[3])
    parser.add_argument('--act', type=str, default=args[4])
    parser.add_argument('--lr', type=float, default=args[5])
    parser.add_argument('--cuda', type=int, default=args[6])
    parser.add_argument('--epochs', type=int, default=args[7])
    parser.add_argument('--seed', type=int, default=args[8])
    parser.add_argument('--eval_freq', type=int, default=args[9])
    parser.add_argument('--val_prop', type=float, default=args[10])
    parser.add_argument('--test_prop', type=float, default=args[11])
    parser.add_argument('--dropout', type=float, default=args[12])
    parser.add_argument('--beta1', type=float, default=args[13])
    parser.add_argument('--beta2', type=float, default=args[14])
    parser.add_argument('--K', type=int, default=args[15])
    parser.add_argument('--beta', type=float, default=args[16])
    parser.add_argument('--analytical_kl', type=bool, default=args[17])
    parser.add_argument('--posterior', type=str, default=args[18])
    parser.add_argument('--prior', type=str, default=args[19])
    parser.add_argument('--prior_iso', type=bool, default=args[20])
    parser.add_argument('--prior_std', type=float, default=args[21])
    parser.add_argument('--learn_prior_std', type=bool, default=args[22])
    parser.add_argument('--enc', type=str, default=args[23])
    parser.add_argument('--dec', type=str, default=args[24])
    parser.add_argument('--bias', type=bool, default=args[25])
    parser.add_argument('--alpha', type=float, default=args[26])
    parser.add_argument('--classifier', type=str, default=args[27])
    parser.add_argument('--clusterer', type=str, default=args[28])
    parser.add_argument('--log_freq', type=int, default=args[29])
    parser.add_argument('--normalize_adj', type=bool, default=args[30])
    parser.add_argument('--normalize_feats', type=bool, default=args[31])
    parser.add_argument('--anomaly_detector', type=str, default=args[32])
    flags, unknown = parser.parse_known_args()
    return flags


def get_activation(args):
    if args.act == 'leaky_relu':
        return nn.LeakyReLU(args.alpha)
    elif args.act == 'rrelu':
        return nn.RReLU()
    elif args.act == 'relu':
        return nn.ReLU()
    elif args.act == 'elu':
        return nn.ELU()
    elif args.act == 'prelu':
        return nn.PReLU()
    elif args.act == 'selu':
        return nn.SELU()


from classifiers import *
def get_classifier(args,X,y):
    if(args.classifier):
        if(args.classifier == 'svm'):
            return SVM(X,y)
        elif(args.classifier == 'mlp'):
            return mlp(X,y,1,10,seed=args.seed)
        elif(args.classifier == 'decision tree'):
            return decision_tree(X,y)
        elif(args.classifier == 'random forest'):
            return random_forest(X,y,args.seed)
        elif(args.classifier == 'adaboost'):
            return adaboost(X,y,args.seed)
        elif(args.classifier == 'knn'):
            return KNN(X,y)
        elif(args.classifier == 'naive bayes'):
            return naive_bayes(X,y)
        else:
            raise NotImplementedError
    

from clusterers import *
def get_clustering_algorithm(clusterer,X,y):
    if(clusterer == 'agglomerative_clustering'):
        return agglomerative_clustering(X,y)
    elif(clusterer == 'dbscan'):
        return dbscan(X,y)
    elif(clusterer == 'fuzzy_c_mean'):
        return fuzzy_c_mean(X,y)
    elif(clusterer == 'gaussian_mixture'):
        return gaussian_mixture(X,y)
    elif(clusterer == 'kmeans'):
        return kmeans(X,y)
    elif(clusterer == 'mean_shift'):
        return mean_shift(X,y)
    else:
        raise NotImplementedError
    
from anomaly_detection import *
def get_anomaly_detection_algorithm(algorithm,X,y):
    if(algorithm == 'isolation_forest'):
        return isolation_forest(X,y)
    elif(algorithm == 'one_class_svm'):
        return one_class_svm(X,y)
    elif(algorithm == 'dbscan'):
        return dbscan(X,y)
    elif(algorithm == 'kmeans'):
        return kmeans(X,y,n_clusters=2)
    elif(algorithm == 'local_outlier_factor'):
        return local_outlier_factor(X,y)
    else:
        raise NotImplementedError