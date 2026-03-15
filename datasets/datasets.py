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
from sklearn.utils import resample
from scipy.io import loadmat
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from scipy.sparse import lil_matrix



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
        with open(self.features_path,'wb') as f:
            pickle.dump(features,f)
        with open(self.labels_path,'wb') as f:
            pickle.dump(labels,f)

    def load_samples(self):
        with open(self.adj_path,'rb') as f:
            adj = pickle.load(f)
        with open(self.features_path,'rb') as f:
            features = pickle.load(f)
        with open(self.labels_path,'rb') as f:
            labels = pickle.load(f)
        print('features:',features.shape,'adj',adj.shape,'labels',labels.shape)
        return adj,features,labels
        
class CIC_DDoS2019(Dataset):
    def __init__(self):
        super().__init__(
            features_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'examples','CIC_DDoS2019','features.pkl'),
            adj_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'examples','CIC_DDoS2019','adjacency.pkl'),
            labels_path= os.path.join(os.path.dirname(os.path.abspath(__file__)),'examples','CIC_DDoS2019','labels.pkl'),
            directory=os.path.join(os.path.dirname(os.path.abspath(__file__)),'examples','CIC_DDoS2019','original')
        )
        self.file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples', 'CIC_DDoS2019', 'original', 'DrDoS_DNS.csv')


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
        
    def _create_file_bc(self, n_nodes, n_classes):
        # Get available files and determine max_per_class
        file_paths = self._get_files()
        max_per_class = int(n_nodes / (n_classes * len(file_paths))) + 1
        df_list = []
    
        for path in file_paths:
            class_name = path.split('/')[-1].split('.')[0]
            list_classes = ['BENIGN', class_name]
            df = self._load_file(path, max_per_class, list_classes)
    
            print(f'finishing loading the file: {path}')
            print(f'Total rows available after filtering: {df.shape[0]}')
    
            # Check if we have enough rows
            if df.shape[0] < n_nodes:
                print(f"Warning: Not enough rows to sample {n_nodes}. Reducing sample size to {df.shape[0]}.")
                n_nodes = df.shape[0]
            
            # Append to list
            df_list.append(df)
    
        # Combine DataFrames and ensure balanced sampling
        df = pd.concat(df_list, ignore_index=True)
        df = df.groupby(' Label', group_keys=False).apply(lambda x: x.sample(min(len(x), n_nodes // n_classes), replace=False))
    
        print(f'DataFrame shape after sampling: {df.shape}')
        print("Class distribution after sampling:")
        print(df[' Label'].value_counts())
    
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
        
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

class Darknet(Dataset):
    def __init__(self):
        super().__init__(
            features_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples', 'Darknet', 'features.pkl'),
            adj_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples', 'Darknet', 'adjacency.pkl'),
            labels_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples', 'Darknet', 'labels.pkl')
        )
        self.file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples', 'Darknet', 'original', 'Darknet.csv')

    def _to_binary_classification(self, x):
        if 'Non' in x:
            return 0
        else:
            return 1

    def build(self, n_nodes, n_classes=2):
        try:
            df = pd.read_csv(self.file, on_bad_lines='skip')
            print("CSV file loaded successfully.", flush=True)
        except pd.errors.ParserError as e:
            print(f"ParserError: {e}")
            return None, None, None       

        df.dropna(axis=0, inplace=True)
        df['Label'] = df['Label'].apply(self._to_binary_classification)
        print(f"DataFrame shape after dropping NaN: {df.shape}", flush=True)

        # Ensure we have n_nodes equally divided across the classes
        df_sampled = df.groupby('Label').apply(lambda x: x.sample(min(len(x), n_nodes // n_classes))).reset_index(drop=True)
        print(f"DataFrame shape after sampling {n_nodes} nodes: {df_sampled.shape}", flush=True)

        # Convert the sampled DataFrame into a numpy array for adjacency matrix creation
        data = df_sampled.to_numpy()

        # Build the full adjacency matrix for the sampled nodes
        full_adj = self._filling_adjacency_numpy(data, source_ip_index=1, destination_ip_index=3)

        # Calculate the degree of each sampled node
        degrees = full_adj.sum(axis=1)

        # Select the top `n_nodes` based on the degree
        selected_indices = np.argsort(degrees)[-n_nodes:]

        # Filter the DataFrame to keep only the selected nodes
        df_selected = df_sampled.iloc[selected_indices]
        labels = df_selected['Label'].to_numpy()

        # Exclude columns that are not part of the features
        columns_to_exclude = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp', 'Label', 'Label.1', 'Protocol', 'Flow Duration']
        features = df_selected.drop(columns=columns_to_exclude).to_numpy()

        # Apply SMOTE for class balancing
        smote = SMOTE(random_state=42)
        features_resampled, labels_resampled = smote.fit_resample(features, labels)
        print(f"Features shape after SMOTE: {features_resampled.shape}, Labels shape after SMOTE: {labels_resampled.shape}", flush=True)

        # Standardize features after resampling
        scaler = StandardScaler()
        features_resampled = scaler.fit_transform(features_resampled)
        print(f"Features shape after scaling: {features_resampled.shape}", flush=True)

        # Filter the adjacency matrix for the selected nodes
        adj = full_adj[np.ix_(selected_indices, selected_indices)]
        print(f"Filtered adjacency matrix shape: {adj.shape}", flush=True)

        print(f"Features shape before saving: {features_resampled.shape}", flush=True)

        self.save_samples(adj, features_resampled, labels_resampled)
        print('Features saved successfully.', flush=True)

        return adj, features_resampled, labels_resampled

    def _filling_adjacency_numpy(self, data, source_ip_index, destination_ip_index):
        N = data.shape[0]
        try:
            adjacency = np.zeros((N, N), dtype=bool)
        except Exception as e:
            print(f"An error occurred: {e}")

        source_ips = data[:, source_ip_index]
        destination_ips = data[:, destination_ip_index]

        # Fill the adjacency matrix based on IP relationships
        mask = (
            (source_ips[:, np.newaxis] == source_ips) |
            (source_ips[:, np.newaxis] == destination_ips) |
            (destination_ips[:, np.newaxis] == source_ips) |
            (destination_ips[:, np.newaxis] == destination_ips)
        )
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


class DGraphFin(Dataset):
    def __init__(self):
        super().__init__(
            features_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples', 'DGraphFin', 'features.pkl'),
            adj_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples', 'DGraphFin', 'adjacency.pkl'),
            labels_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples', 'DGraphFin', 'labels.pkl'),
            directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples', 'DGraphFin', 'original')
        )
        
    def build(self, n_nodes):
        print("Starting to build the dataset...", flush=True)
    
        # Load the dataset from NPZ file
        path = os.path.join(self.directory, 'dgraphfin.npz')
        print(f"Loading dataset from {path}...", flush=True)
        data = np.load(path)
    
        # Extract features and labels from the dataset
        print("Extracting features and labels...", flush=True)
        features = data['x']
        labels = data['y']
    
        # Create a sparse adjacency matrix
        print("Creating sparse adjacency matrix...", flush=True)
        edge_index = data['edge_index']
        num_nodes = features.shape[0]
    
        # Use lil_matrix for efficient sparse matrix creation
        adj = lil_matrix((num_nodes, num_nodes), dtype=bool)
        for edge in edge_index.T:
            adj[edge[0], edge[1]] = True
            adj[edge[1], edge[0]] = True  # Undirected graph
    
        # Convert the matrix to CSR format for optimized calculations
        adj = adj.tocsr()
        print(f"Sparse adjacency matrix created with shape: {adj.shape}", flush=True)
    
        # Convert multiclass labels to binary classification
        print("Converting multiclass labels to binary labels...", flush=True)
        labels = np.where(labels == 1, 1, 0)
    
        # Calculate node degrees (number of connections)
        print("Calculating node degrees...", flush=True)
        degrees = adj.sum(axis=1).A1  # Sum over rows and convert to 1D array
    
        # Sort nodes by degree and select the top `n_nodes`
        print(f"Selecting top {n_nodes} nodes by degree...", flush=True)
        selected_indices = np.argsort(degrees)[-n_nodes:]
    
        # Filter features and labels for selected nodes
        print("Filtering features and labels for selected nodes...", flush=True)
        features_sampled = features[selected_indices]
        labels_sampled = labels[selected_indices]
    
        print("Class distribution after sampling:", flush=True)
        self.print_class_distribution(labels_sampled)
    
        # Apply SMOTE to balance the dataset
        print("Applying SMOTE to balance the dataset...", flush=True)
        smote = SMOTE(random_state=42, k_neighbors=2)
        features_balanced, labels_balanced = smote.fit_resample(features_sampled, labels_sampled)
    
        print("Class distribution after SMOTE:", flush=True)
        self.print_class_distribution(labels_balanced)
    
        # Expand adjacency matrix for new SMOTE nodes (create zero-connection nodes)
        new_node_count = features_balanced.shape[0] - features_sampled.shape[0]
        print(f"Expanding adjacency matrix for {new_node_count} new nodes...", flush=True)
        
        # Create an expanded adjacency matrix with zero connections for the new nodes
        adj_expanded = lil_matrix((features_balanced.shape[0], features_balanced.shape[0]), dtype=bool)
        adj_expanded[:n_nodes, :n_nodes] = adj[selected_indices][:, selected_indices]  # Retain original adj matrix for the first `n_nodes`
        adj_sampled = adj_expanded.toarray()
    
        # Check for NaN values in the features
        if np.isnan(features_balanced).any():
            print("NaN values found in features. Fixing NaN values...", flush=True)
            features_balanced = np.nan_to_num(features_balanced)  # Replace NaNs with 0
        else:
            print("No NaN values found in features.", flush=True)
    
        # Normalize features
        print("Normalizing features...", flush=True)
        scaler = MinMaxScaler()
        features_balanced = scaler.fit_transform(features_balanced)
    
        # Save samples to pickle files
        print("Saving samples to pickle files...", flush=True)
        self.save_samples(adj_sampled, features_balanced, labels_balanced)
    
        print("Dataset build complete.", flush=True)
        return adj_sampled, features_balanced, labels_balanced


    def print_class_distribution(self, labels):
        unique, counts = np.unique(labels, return_counts=True)
        distribution = dict(zip(unique, counts))
        print(f"Class distribution: {distribution}", flush=True)
        for label, count in distribution.items():
            print(f"Class {label}: {count} nodes")

class EllipticDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.features_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples', 'Elliptic', 'features.pkl')
        self.adj_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples', 'Elliptic', 'adjacency.pkl')
        self.labels_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples', 'Elliptic', 'labels.pkl')
        self.directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples', 'Elliptic', 'original')


    def build(self, n_nodes, n_classes=2):
        # Load features and labels
        features_df = pd.read_csv(os.path.join(self.directory, 'txs_features.csv'))
        features_df.columns = ['txId'] + [f'V{i}' for i in range(1, 167)]
    
        labels_df = pd.read_csv(os.path.join(self.directory, 'txs_classes.csv'))
    
        # Map class labels to 'licit' and 'illicit'
        labels_df['class_mapped'] = labels_df['class'].replace({'1': 'illicit', '2': 'licit'})
    
        # Merge features and labels on 'txId'
        merged_df = pd.merge(features_df, labels_df, on='txId')
    
        # Print the class distribution before filtering out unknown classes
        print("Merged Data Class Distribution Before Filtering:\n", merged_df['class'].value_counts())
    
        # Filter out rows with unknown classes
        merged_df = merged_df[merged_df['class_mapped'] != 'unknown']
    
        # Map classes to numerical values
        class_mapping = {
            '2': 1,  # Licit (2 in raw data)
            '1': 0   # Illicit (1 in raw data)
        }
        merged_df['class'] = merged_df['class'].map(class_mapping)
    
        # Print class distribution after filtering and mapping
        print("Class distribution after filtering and mapping:\n", merged_df['class'].value_counts())
    
        # Drop non-numerical columns and prepare features and labels
        features = merged_df.drop(columns=['txId', 'class', 'class_mapped']).to_numpy()  # Drop non-numerical columns
        labels = merged_df['class'].to_numpy()
    
        # Normalize the features (fixed the error of string to float conversion)
        scaler = MinMaxScaler()
        features = scaler.fit_transform(features)
    
        # Convert features to a DataFrame for easier sampling
        df_combined = pd.DataFrame(features)
        df_combined['Label'] = labels
    
        # Get the counts of each class
        licit_count = len(df_combined[df_combined['Label'] == 1])
        illicit_count = len(df_combined[df_combined['Label'] == 0])
    
        # Adjust the number of samples per class
        illicit_sample_count = min(illicit_count, n_nodes // n_classes)  # Max possible illicit samples
        licit_sample_count = n_nodes - illicit_sample_count  # Remaining should be licit
    
        # Sample illicit and licit rows
        illicit_samples = df_combined[df_combined['Label'] == 0].sample(n=illicit_sample_count, random_state=42)
        licit_samples = df_combined[df_combined['Label'] == 1].sample(n=licit_sample_count, random_state=42)
    
        # Concatenate the sampled rows
        df_sampled = pd.concat([illicit_samples, licit_samples], ignore_index=True)
    
        # Print the balanced class distribution after sampling
        print("Balanced class distribution after sampling:\n", df_sampled['Label'].value_counts())
    
        # Convert the sampled data to features and labels
        features_sampled = df_sampled.drop(columns=['Label']).to_numpy()
        labels_sampled = df_sampled['Label'].to_numpy()
    
        # Build adjacency matrix
        # Load the edge list
        edge_list_df = pd.read_csv(os.path.join(self.directory, 'txs_edgelist.csv'))
        source = edge_list_df['txId1'].to_numpy()
        destination = edge_list_df['txId2'].to_numpy()
    
        # Map node ids to indices
        unique_nodes = np.unique(np.concatenate((source, destination)))
        node_map = {node: idx for idx, node in enumerate(unique_nodes)}
    
        # Create the adjacency matrix
        N = len(unique_nodes)
        adj = np.zeros((N, N), dtype=bool)
        for src, dest in zip(source, destination):
            if src in node_map and dest in node_map:
                adj[node_map[src], node_map[dest]] = True
                adj[node_map[dest], node_map[src]] = True 
    
        # Build the full adjacency matrix for sampled nodes
        sampled_node_ids = df_sampled.index.to_numpy()
        sampled_adj = adj[np.ix_(sampled_node_ids, sampled_node_ids)]
    
        # Optional: Select based on highest degree
        degrees = sampled_adj.sum(axis=1)
        selected_indices = np.argsort(degrees)[-n_nodes:]
    
        # Select features, labels, and adjacency matrix based on selected nodes
        features_selected = features_sampled[selected_indices]
        labels_selected = labels_sampled[selected_indices]
        adj_selected = sampled_adj[np.ix_(selected_indices, selected_indices)]
    
        return adj_selected, features_selected, labels_selected
    
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

    def save_samples(self, adj, features, labels):
        with open(self.adj_path, 'wb') as f:
            pickle.dump(adj, f)
        with open(self.features_path, 'wb') as f:
            pickle.dump(features, f)
        with open(self.labels_path, 'wb') as f:
            pickle.dump(labels, f)
    def __init__(self):
        super().__init__(
            features_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples', 'Cora', 'features.pkl'),
            adj_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples', 'Cora', 'adjacency.pkl'),
            labels_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples', 'Cora', 'labels.pkl'),
            directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples', 'Cora', 'original')
        )

    def load_data(self):
        # Load node data (features + labels)
        feature_names = ["w_{}".format(ii) for ii in range(1433)]
        column_names = feature_names + ["subject"]
        node_data = pd.read_csv(os.path.join(self.directory, 'cora.content'), sep='\t', names=column_names)
        
        # Load edge list (source-target pairs)
        edge_list = pd.read_csv(os.path.join(self.directory, 'cora.cites'), sep='\t', names=['target', 'source'])

        # Node and feature statistics
        num_node, num_feature = node_data.shape[0], node_data.shape[1] - 1
        print(f"The number of nodes: {num_node}")
        print(f"The number of features: {num_feature}")
        print(f"The number of edges: {edge_list.shape[0]}")

        # Create binary classification label: 'anomaly' for specified subjects, 'normal' otherwise
        node_data['label'] = node_data['subject'].apply(
            lambda x: 'anomaly' if x in ['Neural_Networks', 'Rule_Learning', 'Probabilistic_Methods'] else 'normal'
        )
        print(node_data['label'].value_counts())

        # Check for and handle missing values
        if node_data.isnull().any().any():
            print("Missing values detected in the dataset. Filling missing values with 0.")
            node_data.fillna(0, inplace=True)  # Alternatively, use mean/median imputation
            # Example using mean imputation:
            # imputer = SimpleImputer(strategy='mean')
            # node_data.iloc[:, :-2] = imputer.fit_transform(node_data.iloc[:, :-2])

        return node_data, edge_list

    def build(self):
        # Load the data
        node_data, edge_list = self.load_data()

        # Get adjacency matrix
        adj = self.build_adjacency_matrix(edge_list, node_data)

        # Get features and labels
        features = node_data.iloc[:, :-2].values  # Exclude 'subject' and 'label' columns
        labels = (node_data['label'] == 'anomaly').astype(int).values  # Binary labels: 1 for anomaly, 0 for normal

        # Check for finite values before scaling
        if not np.all(np.isfinite(features)):
            print("Non-finite values detected in features before scaling. Replacing with 0.")
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Standardize the features
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


        # Check for NaNs or infinities after scaling
        if np.isnan(features).any() or np.isinf(features).any():
            print("NaNs or infinities detected in features after scaling. Replacing with 0.")
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Check for zero variance features and remove them
        variances = np.var(features, axis=0)
        zero_var = variances == 0
        if zero_var.any():
            zero_var_indices = np.where(zero_var)[0]
            print(f"Features with zero variance detected at indices: {zero_var_indices}. Removing these features.")
            features = features[:, ~zero_var]

        
        # Save adjacency, features, and labels
        self.save_samples(adj=adj, features=features, labels=labels)

        return adj, features, labels

    def build_adjacency_matrix(self, edge_list, node_data):
        num_nodes = node_data.shape[0]
        adjacency = np.zeros((num_nodes, num_nodes), dtype=bool)

        # Create a mapping between node IDs and row indices in the node_data
        node_map = {node: idx for idx, node in enumerate(node_data.index)}

        # Fill the adjacency matrix based on the edge list
        for _, edge in edge_list.iterrows():
            if edge['source'] in node_map and edge['target'] in node_map:
                source_idx = node_map[edge['source']]
                target_idx = node_map[edge['target']]
                adjacency[source_idx, target_idx] = True
                adjacency[target_idx, source_idx] = True

        # Make sure the diagonal is set to True (self-loops)
        np.fill_diagonal(adjacency, True)

        return adjacency

    def save_samples(self, adj, features, labels):
        # Save adjacency, features, and labels as .pkl files
        pd.to_pickle(adj, self.adj_path)
        pd.to_pickle(features, self.features_path)
        pd.to_pickle(labels, self.labels_path)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter

class AmazonDataset:
    def __init__(self):
        # Define paths to the dataset files
        self.directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples', 'Amazon', 'original')

    def build(self):
        # Load Amazon dataset from .mat file
        mat = loadmat(os.path.join(self.directory, 'Amazon.mat'))
    
        # Extract features and labels from the .mat file
        features = mat['X']  # Features
        labels = mat['gnd'].flatten()  # Labels
        
        # Check if adjacency matrix 'A' is a sparse matrix
        if 'A' in mat:
            adj = mat['A']
            if hasattr(adj, 'toarray'):
                adj = adj.toarray()  # Convert to dense if it's sparse
        else:
            raise ValueError("Adjacency matrix 'A' not found in the .mat file.")
    
        # Normalize features (standardization)
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        # Class distribution before balancing
        print("Class distribution before balancing:")
        self.print_class_distribution(labels)

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels, random_state=42)

        # Apply SMOTE to the training set
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # Print class distribution after balancing
        print("Class distribution after balancing:")
        self.print_class_distribution(y_train_resampled)

        # Ensure adjacency and features have the same number of nodes
        if adj.shape[0] != features.shape[0]:
            raise ValueError(f"Adjacency matrix and features size mismatch! adj.shape: {adj.shape}, features.shape: {features.shape}")

        # Return the adjacency matrix, features, and labels
        return adj, features, labels
        
    def print_class_distribution(self, labels):
        # Count the occurrences of each class
        class_counts = Counter(labels)

        # Print the class distribution
        print("Class distribution:")
        for cls, count in class_counts.items():
            print(f"Class {cls}: {count} instances")

from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
from scipy.io import loadmat

class YelpNYCDataset:
    def __init__(self, n_nodes=2000, n_classes=2):
        # Define paths to the dataset files
        self.directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples', 'YelpNYC')
        self.n_nodes = n_nodes
        self.n_classes = n_classes

        # Paths for saving .pkl files
        self.features_path = os.path.join(self.directory, 'features.pkl')
        self.labels_path = os.path.join(self.directory, 'labels.pkl')
        self.adj_path = os.path.join(self.directory, 'adjacency.pkl')

    def build(self):
        # Load YelpNYC dataset from .mat file
        mat = loadmat(os.path.join(self.directory, 'YelpNYC.mat'))

        # Extract features, adjacency matrix, and labels from the .mat file
        features = mat['Attributes']  # Assuming 'Attributes' contains the node features
        labels = mat['Label'].flatten()  # Assuming 'Label' contains the node labels

        # Check if features is a sparse matrix and convert to dense format if necessary
        if sp.issparse(features):
            features = features.toarray()  # Convert to dense format
        else:
            features = np.array(features)  # Convert to array if already dense

        # Ensure features are float32
        features = np.array(features, dtype=np.float32)

        # Check if adjacency matrix 'Network' is a sparse matrix
        adj = mat['Network']  # Assuming 'Network' contains the adjacency matrix
        if hasattr(adj, 'toarray'):
            adj = adj.toarray()  # Convert to dense if it's sparse

        # Normalize features (standardization), ensuring sparse data is handled
        scaler = StandardScaler(with_mean=False)  # Avoid centering for sparse matrices
        features = scaler.fit_transform(features)

        # Check for NaN values in features and remove them if necessary
        if np.isnan(features).any():
            valid_indices = ~np.isnan(features).any(axis=1)
            features = features[valid_indices]
            labels = labels[valid_indices]  # Update labels accordingly
            adj = adj[valid_indices, :][:, valid_indices]  # Keep only valid rows and columns

        # Ensure that the number of nodes is still sufficient after removing NaNs
        if features.shape[0] < self.n_nodes:
            raise ValueError(f"Not enough nodes after removing NaN values. Available: {features.shape[0]}, Required: {self.n_nodes}")

        # Perform degree-based sampling
        adj_sampled, features_sampled, labels_sampled = self.degree_based_sampling(adj, features, labels)

        # Save adjacency, features, and labels
        self.save_samples(adj_sampled, features_sampled, labels_sampled)

        return adj_sampled, features_sampled, labels_sampled

    def degree_based_sampling(self, adj, features, labels):
        # Calculate the degree of each node
        if sp.issparse(adj):
            degrees = adj.sum(axis=1).A1  # Convert to 1D array
        else:
            degrees = adj.sum(axis=1)  # Directly sum if it's a dense array

        # Sample nodes ensuring balance across classes
        sampled_indices = []
        unique_labels = np.unique(labels)

        for label in unique_labels:
            class_indices = np.where(labels == label)[0]
            num_samples = min(len(class_indices), self.n_nodes // self.n_classes)
            sampled_class_indices = np.random.choice(class_indices, num_samples, replace=False)
            sampled_indices.extend(sampled_class_indices)

        sampled_indices = np.array(sampled_indices)

        features_sampled = features[sampled_indices]
        labels_sampled = labels[sampled_indices]
        adj_sampled = adj[np.ix_(sampled_indices, sampled_indices)]

        return adj_sampled, features_sampled, labels_sampled

    def save_samples(self, adj, features, labels):
        # Save adjacency, features, and labels as .pkl files
        pd.to_pickle(adj, self.adj_path)
        pd.to_pickle(features, self.features_path)
        pd.to_pickle(labels, self.labels_path)

        print(f"Saved adjacency matrix to {self.adj_path}")
        print(f"Saved features to {self.features_path}")
        print(f"Saved labels to {self.labels_path}")

class YelpHotelDataset:
    def __init__(self, n_nodes=1000):
        # Define paths to the dataset files
        self.directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples', 'YelpHotel')
        self.n_nodes = n_nodes

    def build(self):
        # Load YelpHotel dataset from .mat file
        mat = loadmat(os.path.join(self.directory, 'YelpHotel.mat'))

        # Extract features, adjacency matrix, and labels from the .mat file
        features = mat['Attributes']  # Assuming 'Attributes' contains the node features
        labels = mat['Label'].flatten()  # Assuming 'Label' contains the node labels

        # Print out the type of features
        print(f"Original features type: {type(features)}")

        # Check if features is a sparse matrix and convert to dense format if necessary
        if sp.issparse(features):
            features = features.toarray()  # Convert to dense format
        else:
            features = np.array(features)  # Convert to array if already dense

        # Ensure features are float32
        features = np.array(features, dtype=np.float32)

        # Check if adjacency matrix 'Network' is a sparse matrix
        adj = mat['Network']  # Assuming 'Network' contains the adjacency matrix
        if hasattr(adj, 'toarray'):
            adj = adj.toarray()  # Convert to dense if it's sparse

        # Normalize features (standardization), ensuring sparse data is handled
        scaler = StandardScaler(with_mean=False)  # Avoid centering for sparse matrices
        features = scaler.fit_transform(features)

        # Check for NaN values in features
        print(f"Features shape after scaling: {features.shape}")
        print(f"NaN values present: {np.isnan(features).any()}")  # Debugging statement

        # If NaN values are present, remove them
        if np.isnan(features).any():
            print("NaN values found in features. Removing rows with NaN values.")
            valid_indices = ~np.isnan(features).any(axis=1)
            features = features[valid_indices]
            labels = labels[valid_indices]  # Update labels accordingly

            # Also adjust the adjacency matrix based on the valid indices
            adj = adj[valid_indices, :][:, valid_indices]  # Keep only valid rows and columns

        # Ensure that the number of nodes is still sufficient after removing NaNs
        if features.shape[0] < self.n_nodes:
            raise ValueError(f"Not enough nodes after removing NaN values. Available: {features.shape[0]}, Required: {self.n_nodes}")

        # Print class distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        print("Class distribution:")
        for label, count in zip(unique_labels, counts):
            print(f'Label {label}: {count} instances')

        # Return the adjacency matrix, features, and labels
        return adj, features, labels
