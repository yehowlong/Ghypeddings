import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import pickle
import time
import datetime
import progressbar
import category_encoders as ce
from sklearn.utils import resample
from scipy.io import loadmat
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from scipy.sparse import lil_matrix, issparse
import scipy.sparse as sp
from collections import Counter

class Dataset:
    def __init__(self, features_path='', adj_path='', labels_path='', directory=''):
        self.features_path = features_path
        self.adj_path = adj_path
        self.labels_path = labels_path
        self.directory = directory

    def _get_files(self):
        return [os.path.join(self.directory, file) for file in os.listdir(self.directory)
                if os.path.isfile(os.path.join(self.directory, file)) and '.gitignore' not in file]

    def save_samples(self, adj, features, labels):
        with open(self.adj_path, 'wb') as f:
            pickle.dump(adj, f)
        with open(self.features_path, 'wb') as f:
            pickle.dump(features, f)
        with open(self.labels_path, 'wb') as f:
            pickle.dump(labels, f)

    def load_samples(self):
        with open(self.adj_path, 'rb') as f:
            adj = pickle.load(f)
        with open(self.features_path, 'rb') as f:
            features = pickle.load(f)
        with open(self.labels_path, 'rb') as f:
            labels = pickle.load(f)
        print('features:', features.shape, 'adj', adj.shape, 'labels', labels.shape)
        return adj, features, labels

# --- 其他数据集类 (CIC_DDoS2019, NetFlowDataset, NF_CIC_IDS2018_v2, NF_UNSW_NB15_v2, Darknet, NF_BOT_IoT_v2, NF_TON_IoT_v2, AWID3, DGraphFin, EllipticDataset, AmazonDataset, YelpNYCDataset, YelpHotelDataset) ---
# 注意：这里为了篇幅省略了中间未改动的类，请保留你原始文件中这些类的完整定义

# ... [保留中间的所有类定义] ...

class YelpHotelDataset(Dataset): # 修正后的 YelpHotelDataset
    def __init__(self, n_nodes=1000):
        self.directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples', 'YelpHotel')
        self.n_nodes = n_nodes
        super().__init__()

    def build(self):
        mat = loadmat(os.path.join(self.directory, 'YelpHotel.mat'))
        features = mat['Attributes']
        labels = mat['Label'].flatten()
        if sp.issparse(features):
            features = features.toarray()
        features = np.array(features, dtype=np.float32)
        adj = mat['Network']
        if hasattr(adj, 'toarray'):
            adj = adj.toarray()
        scaler = StandardScaler(with_mean=False)
        features = scaler.fit_transform(features)
        if np.isnan(features).any():
            valid_indices = ~np.isnan(features).any(axis=1)
            features = features[valid_indices]
            labels = labels[valid_indices]
            adj = adj[valid_indices, :][:, valid_indices]
        if features.shape[0] < self.n_nodes:
            raise ValueError(f"Not enough nodes. Available: {features.shape[0]}")
        return adj, features, labels

# ✨ 重点：修复 Cora 类的定义 (之前这里漏掉了 class 关键字且缩进错误)
class Cora(Dataset):
    def __init__(self):
        super().__init__(
            features_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples', 'Cora', 'features.pkl'),
            adj_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples', 'Cora', 'adjacency.pkl'),
            labels_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples', 'Cora', 'labels.pkl'),
            directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples', 'Cora', 'original')
        )

    def load_data(self):
        feature_names = ["w_{}".format(ii) for ii in range(1433)]
        column_names = feature_names + ["subject"]
        node_data = pd.read_csv(os.path.join(self.directory, 'cora.content'), sep='\t', names=column_names)
        edge_list = pd.read_csv(os.path.join(self.directory, 'cora.cites'), sep='\t', names=['target', 'source'])
        node_data['label'] = node_data['subject'].apply(
            lambda x: 'anomaly' if x in ['Neural_Networks', 'Rule_Learning', 'Probabilistic_Methods'] else 'normal'
        )
        return node_data, edge_list

    def build(self):
        node_data, edge_list = self.load_data()
        adj = self.build_adjacency_matrix(edge_list, node_data)
        features = node_data.iloc[:, :-2].values
        labels = (node_data['label'] == 'anomaly').astype(int).values
        if not np.all(np.isfinite(features)):
            features = np.nan_to_num(features)
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        self.save_samples(adj=adj, features=features, labels=labels)
        return adj, features, labels

    def build_adjacency_matrix(self, edge_list, node_data):
        num_nodes = node_data.shape[0]
        adjacency = np.zeros((num_nodes, num_nodes), dtype=bool)
        node_map = {node: idx for idx, node in enumerate(node_data.index)}
        for _, edge in edge_list.iterrows():
            if edge['source'] in node_map and edge['target'] in node_map:
                s, t = node_map[edge['source']], node_map[edge['target']]
                adjacency[s, t] = adjacency[t, s] = True
        np.fill_diagonal(adjacency, True)
        return adjacency

    def save_samples(self, adj, features, labels):
        pd.to_pickle(adj, self.adj_path)
        pd.to_pickle(features, self.features_path)
        pd.to_pickle(labels, self.labels_path)