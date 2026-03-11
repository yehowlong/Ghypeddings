import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.preprocessing import LabelEncoder
import time
import datetime
import progressbar
import category_encoders as ce

class Dataset:
    def __init__(self,features_path='',adj_path='',labels_path='',directory=''):
        self.features_path = features_path
        self.adj_path = adj_path
        self.labels_path = labels_path
        self.directory = directory

    def _get_files(self):
        return [os.path.join(self.directory,file) for file in os.listdir(self.directory) if os.path.isfile(os.path.join(self.directory, file)) and '.gitignore' not in file]

    def save_samples(self,adj,features,labels):
        with open(self.adj_path,'wb') as f:
            pickle.dump(adj,f)
        print('The adjacency matrix is saved in',self.adj_path)
        with open(self.features_path,'wb') as f:
            pickle.dump(features,f)
        print('The node features matrix is saved in',self.features_path)
        with open(self.labels_path,'wb') as f:
            pickle.dump(labels,f)
        print('The labels are saved in ',self.labels_path)

    def load_samples(self):
        with open(self.adj_path,'rb') as f:
            adj = pickle.load(f)
        print('The adjacency matrix has been loaded successfully')
        with open(self.features_path,'rb') as f:
            features = pickle.load(f)
        print('The node features matrix has been loaded successfully')
        with open(self.labels_path,'rb') as f:
            labels = pickle.load(f)
        print('The labels have been loaded successfully')
        print('features shape:',features.shape,'adj shape',adj.shape,'labels shape',labels.shape)
        return adj,features,labels

class CIC_DDoS2019(Dataset):
    def __init__(self):
        super().__init__(
            features_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'examples','CICDDoS2019','features.pkl'),
            adj_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'examples','CICDDoS2019','adjacency.pkl'),
            labels_path= os.path.join(os.path.dirname(os.path.abspath(__file__)),'examples','CICDDoS2019','labels.pkl'),
            directory=os.path.join(os.path.dirname(os.path.abspath(__file__)),'examples','CICDDoS2019','original')
        )

    def build(self,n_nodes,n_classes=2):
        df = self._create_file_bc(n_nodes,n_classes)
        for column in df.columns:
            max_value = df.loc[df[column] != np.inf, column].max()
            min_value = df.loc[df[column] != -np.inf, column].min()
            df.loc[df[column] == np.inf, column] = max_value
            df.loc[df[column] == -np.inf, column] = min_value
        adj = self._filling_adjacency_numpy(df)
        labels = df[' Label'].apply(lambda x: 0 if x == 'BENIGN' else 1).to_numpy()
        columns_to_exclude = ['Unnamed: 0', 'Flow ID', ' Source IP',' Source Port',' Destination Port',' Flow Duration',' Protocol', ' Destination IP', ' Timestamp', 'SimillarHTTP',' Inbound',' Label']
        df.drop(columns_to_exclude, axis=1, inplace=True)
        features = df.to_numpy()
        scaler = MinMaxScaler()
        features = scaler.fit_transform(features)
        self.save_samples(adj,features,labels)
        return adj, features, labels
    
    def _load_file(self,path,max_per_class,list_classes=[]):
        df = pd.read_csv(path,low_memory=False)
        df.dropna(axis=0, inplace=True)
        if(len(list_classes)):
            df = df[df[' Label'].isin(list_classes)]
            df = df.groupby([' Label']).apply(lambda x: x.sample(max_per_class)).reset_index(drop=True)
        return df
        
    def _create_file_bc(self,n_nodes,n_classes):
        file_paths = self._get_files()
        max_per_class = int(n_nodes / (n_classes * len(file_paths))) +1
        df_list = []
        for path in file_paths:
            class_name = path.split('\\')[-1].split('.')[0]
            list_classes = ['BENIGN',class_name]
            df_list.append(self._load_file(path,max_per_class,list_classes))
            print('finishing loading the file : {}'.format(path))
        df = pd.concat(df_list,ignore_index=True)
        df = df.sample(n=n_nodes).reset_index(drop=True)
        print(df.shape)
        # print(df[' Label'].value_counts())
        # df = pd.read_csv(os.path.join(self.directory,'all.csv'),low_memory=False)
        # df[' Label'] = df[' Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
        # node_per_class = int(n_nodes/n_classes)
        # df = df.groupby([' Label']).apply(lambda x: x.sample(node_per_class)).reset_index(drop=True)
        return df

    def _filling_adjacency_numpy(self,data):
        N = data.shape[0]
        try:
            adjacency = np.zeros((N,N), dtype=bool)
        except Exception as e:
            print(f"An error occurred: {e}")

        source_ips = data[' Source IP'].to_numpy()
        destination_ips = data[' Destination IP'].to_numpy()
        mask = ((source_ips[:, np.newaxis] == source_ips) | (source_ips[:, np.newaxis] == destination_ips) | (destination_ips[:, np.newaxis] == source_ips)| (destination_ips[:, np.newaxis] == destination_ips) )
        adjacency[mask] = True
        return adjacency

class NetFlowDataset(Dataset):
    def __init__(self,features_path,adj_path,labels_path,file):
        super().__init__(features_path,adj_path,labels_path)
        self.file = file

    def build(self,n_nodes,n_classes=2):
        df = pd.read_csv(self.file) 
        df = df.groupby(['Label']).apply(lambda x: x.sample(int(n_nodes/n_classes))).reset_index(drop=True) 
        df = df.sample(frac=1).reset_index(drop=True)
        adj = self._filling_adjacency_numpy(df)
        labels = df['Label'].to_numpy()
        labels = labels.astype(np.bool_)
        df.drop(['IPV4_SRC_ADDR','IPV4_DST_ADDR','Attack','Label','L4_SRC_PORT','L4_DST_PORT'],axis=1,inplace=True)
        #df = pd.get_dummies(df,columns=['PROTOCOL','DNS_QUERY_TYPE','FTP_COMMAND_RET_CODE'])

        encoder = ce.TargetEncoder(cols=['TCP_FLAGS','L7_PROTO','PROTOCOL'])
        encoder.fit(df,labels)
        df = encoder.transform(df)
 
        features = df.to_numpy()
        scaler = MinMaxScaler()
        features = scaler.fit_transform(features)
        print("features:",features.shape)
        self.save_samples(adj,features,labels)
        return adj,features,labels

    def _filling_adjacency_numpy(self,data):
        N = data.shape[0]
        try:
            adjacency = np.zeros((N,N), dtype=bool)
        except Exception as e:
            print(f"An error occurred: {e}")

        if 'bot_iot' in self.file:
            data['IPV4_SRC_ADDR'] = data['IPV4_SRC_ADDR'].apply(str)
            data['IPV4_DST_ADDR'] = data['IPV4_DST_ADDR'].apply(str)
            data['L4_SRC_PORT'] = data['L4_SRC_PORT'].apply(str)
            data['L4_DST_PORT'] = data['L4_DST_PORT'].apply(str)
            data['IPV4_SRC_ADDR'] = data['IPV4_SRC_ADDR']+':'+data['L4_SRC_PORT']
            data['IPV4_DST_ADDR'] = data['IPV4_DST_ADDR']+':'+data['L4_DST_PORT']

        source_ips = data['IPV4_SRC_ADDR'].to_numpy()
        destination_ips = data['IPV4_DST_ADDR'].to_numpy()
        mask = ((source_ips[:, np.newaxis] == source_ips) | (source_ips[:, np.newaxis] == destination_ips) | (destination_ips[:, np.newaxis] == source_ips) | (destination_ips[:, np.newaxis] == destination_ips))
        adjacency[mask] = True
        return adjacency

class NF_CIC_IDS2018_v2(NetFlowDataset):
    def __init__(self):
        super().__init__(
            features_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'examples','CIC_IDS2018','features.pkl'),
            adj_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'examples','CIC_IDS2018','adjacency.pkl'),
            labels_path= os.path.join(os.path.dirname(os.path.abspath(__file__)),'examples','CIC_IDS2018','labels.pkl'),
            file = os.path.join(os.path.dirname(os.path.abspath(__file__)),'examples','CIC_IDS2018','original','cic_ids2018.csv')
        )   

class NF_UNSW_NB15_v2(NetFlowDataset):
    def __init__(self):
        super().__init__(
            features_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'examples','UNSW_NB15','features.pkl'),
            adj_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'examples','UNSW_NB15','adjacency.pkl'),
            labels_path= os.path.join(os.path.dirname(os.path.abspath(__file__)),'examples','UNSW_NB15','labels.pkl'),
            file = os.path.join(os.path.dirname(os.path.abspath(__file__)),'examples','UNSW_NB15','original','unsw_nb15.csv')
        )

class Darknet(Dataset):
    def __init__(self):
        super().__init__(
            features_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'examples','Darknet','features.pkl'),
            adj_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'examples','Darknet','adjacency.pkl'),
            labels_path= os.path.join(os.path.dirname(os.path.abspath(__file__)),'examples','Darknet','labels.pkl')
        )
        self.file = os.path.join(os.path.dirname(os.path.abspath(__file__)),'examples','Darknet','original','Darknet.csv')

    def _to_binary_classification(self,x):
        if 'Non' in x:
            return 0
        else:
            return 1

    def build(self,n_nodes,n_classes=2):
        print('Starting building a graph of size ...',n_nodes)
        df = pd.read_csv(self.file)
        df.dropna(axis=0, inplace=True)
        df['Label'] = df['Label'].apply(self._to_binary_classification)
        df = df.groupby(['Label']).apply(lambda x: x.sample(int(n_nodes/n_classes))).reset_index(drop=True)
        df = df.sample(n=n_nodes).reset_index(drop=True)
        data = df.to_numpy()
        print('finishing data preprocessing ...')
        adj = self._filling_adjacency_numpy(data,1,3)
        print('building the adjacency matrix ...')
        labels = df['Label'].to_numpy()
        columns_to_exclude = ['Flow ID', 'Src IP','Src Port', 'Dst IP','Dst Port', 'Timestamp','Label','Label.1','Protocol','Flow Duration']
        df.drop(columns_to_exclude, axis=1, inplace=True)
        features = df.to_numpy()
        print('saving the graph in the current project ...')
        self.save_samples(adj,features,labels)
        print('Building a graph has been successfully finished !')
        return adj,features,labels
    
    def _filling_adjacency_numpy(self,data,source_ip_index, destination_ip_index):
        N = data.shape[0]
        try:
            adjacency = np.zeros((N,N), dtype=bool)
        except Exception as e:
            print(f"An error occurred: {e}")
        source_ips = data[:, source_ip_index]
        destination_ips = data[:, destination_ip_index]
        mask = ((source_ips[:, np.newaxis] == source_ips) | (source_ips[:, np.newaxis] == destination_ips) | (destination_ips[:, np.newaxis] == source_ips) | (destination_ips[:, np.newaxis] == destination_ips))
        adjacency[mask] = True
        return adjacency

class NF_BOT_IoT_v2(NetFlowDataset):
    def __init__(self):
        super().__init__(
            features_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'examples','BOT_IOT','features.pkl'),
            adj_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'examples','BOT_IOT','adjacency.pkl'),
            labels_path= os.path.join(os.path.dirname(os.path.abspath(__file__)),'examples','BOT_IOT','labels.pkl'),
            file = os.path.join(os.path.dirname(os.path.abspath(__file__)),'examples','BOT_IOT','original','bot_iot.csv')
        )

class NF_TON_IoT_v2(NetFlowDataset):
    def __init__(self):
        # directory=os.path.join(os.path.dirname(os.path.abspath(__file__)),'examples','TON_IOT','original'),
        super().__init__(
            features_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'examples','TON_IOT','features.pkl'),
            adj_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'examples','TON_IOT','adjacency.pkl'),
            labels_path= os.path.join(os.path.dirname(os.path.abspath(__file__)),'examples','TON_IOT','labels.pkl'),
            file = os.path.join(os.path.dirname(os.path.abspath(__file__)),'examples','TON_IOT','original','ton_iot.csv')
        )

class AWID3(Dataset):
    def __init__(self):
        super().__init__(
            features_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'examples','AWID3','features.pkl'),
            adj_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'examples','AWID3','adjacency.pkl'),
            labels_path= os.path.join(os.path.dirname(os.path.abspath(__file__)),'examples','AWID3','labels.pkl'),
            directory=os.path.join(os.path.dirname(os.path.abspath(__file__)),'examples','AWID3','original')
        )

    def _config_signal(self,x):
        words = str(x).split('-')
        return np.mean([float(i)*-1 for i in words if i!=''])
    
    def build(self,n_nodes):
        path = os.path.join(os.getcwd(),'Ghypeddings','datasets','examples','AWID3','original','awid3.csv')
        df = pd.read_csv(path)
        df['Label'] = df['Label'].apply(lambda x: 0 if 'Normal' in x else 1)
        df = df.groupby(['Label']).apply(lambda x: x.sample(int(n_nodes/2))).reset_index(drop=True)
        df = df.sample(frac=1).reset_index(drop=True)
        data=df[['ip.src','ip.dst']]
        df.dropna(axis=1, inplace=True)
        to_drop = ['frame.number','frame.time','radiotap.timestamp.ts','frame.time_delta_displayed','frame.time_epoch','frame.time_relative','wlan.duration','wlan.ra']
        df.drop(columns=to_drop,axis=1,inplace=True)
        alone = []
        for c in df.columns:
            if(len(df[c].unique()) == 1):
                alone.append(c)
            elif len(df[c].unique()) == 2:
                df = pd.get_dummies(df,columns=[c],drop_first=True)
            elif len(df[c].unique()) <=8:
                df = pd.get_dummies(df,columns=[c])
            elif len(df[c].unique()) <=15:
                labels = df['Label']
                df.drop(columns=['Label'],axis=1,inplace=True)
                encoder = ce.TargetEncoder(cols=[c])
                encoder.fit(df,labels)
                df = encoder.transform(df)
                df['Label']=labels
            else:
                if(df[c].dtype == 'object' and c!='radiotap.dbm_antsignal'):
                    print(c,df[c].unique(),len(df[c].unique()))
        df.drop(columns=alone,axis=1,inplace=True)
        df['radiotap.dbm_antsignal'] = df['radiotap.dbm_antsignal'].apply(self._config_signal) # It contains a list
        labels = df['Label_1'].to_numpy()
        adj = self._filling_adjacency_numpy(data)
        df.drop(columns=['frame.time_delta','Label_1'],axis=1,inplace=True)
        features = df.to_numpy()
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        # scaler = MinMaxScaler()
        # features = scaler.fit_transform(features)
        self.save_samples(adj=adj,features=features,labels=labels)
        return adj,features,labels
    
    def _filling_adjacency_numpy(self,data):
        N = data.shape[0]
        try:
            adjacency = np.zeros((N,N), dtype=bool)
        except Exception as e:
            print(f"An error occurred: {e}")
        source_ips = data['ip.src'].to_numpy()
        destination_ips = data['ip.dst'].to_numpy()
        mask = ((source_ips[:, np.newaxis] == source_ips) | (source_ips[:, np.newaxis] == destination_ips) | (destination_ips[:, np.newaxis] == source_ips) | (destination_ips[:, np.newaxis] == destination_ips) )
        adjacency[mask] = True
        np.fill_diagonal(adjacency, True)
        return adjacency