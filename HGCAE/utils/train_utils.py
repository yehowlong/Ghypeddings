import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.modules.loss
import argparse

def format_metrics(metrics, split):
    """Format metric in metric dict for logging."""
    return " ".join(
            ["{}_{}: {:.8f}".format(split, metric_name, metric_val) for metric_name, metric_val in metrics.items()])

def get_dir_name(models_dir):
    """Gets a directory to save the model.

    If the directory already exists, then append a new integer to the end of
    it. This method is useful so that we don't overwrite existing models
    when launching new jobs.

    Args:
        models_dir: The directory where all the models are.

    Returns:
        The name of a new directory to save the training logs and model weights.
    """
    if not os.path.exists(models_dir):
        save_dir = os.path.join(models_dir, '0')
        os.makedirs(save_dir)
    else:
        existing_dirs = np.array(
                [
                    d
                    for d in os.listdir(models_dir)
                    if os.path.isdir(os.path.join(models_dir, d))
                    ]
        ).astype(np.int)
        if len(existing_dirs) > 0:
            dir_id = str(existing_dirs.max() + 1)
        else:
            dir_id = "1"
        save_dir = os.path.join(models_dir, dir_id)
        os.makedirs(save_dir)
    return save_dir


def add_flags_from_config(parser, config_dict):
    """
    Adds a flag (and default value) to an ArgumentParser for each parameter in a config
    """

    def OrNone(default):
        def func(x):
            # Convert "none" to proper None object
            if x.lower() == "none":
                return None
            # If default is None (and x is not None), return x without conversion as str
            elif default is None:
                return str(x)
            # Otherwise, default has non-None type; convert x to that type
            else:
                return type(default)(x)

        return func

    for param in config_dict:
        default, description = config_dict[param]
        try:
            if isinstance(default, dict):
                parser = add_flags_from_config(parser, default)
            elif isinstance(default, list):
                if len(default) > 0:
                    # pass a list as argument
                    parser.add_argument(
                            f"--{param}",
                            action="append",
                            type=type(default[0]),
                            default=default,
                            help=description
                    )
                else:
                    pass
                    parser.add_argument(f"--{param}", action="append", default=default, help=description)
            else:
                pass
                parser.add_argument(f"--{param}", type=OrNone(default), default=default, help=description)
        except argparse.ArgumentError:
            print(
                f"Could not add flag for param {param} because it was already present."
            )
    return parser



import subprocess
def check_gpustats(columns=None):
    query = r'nvidia-smi --query-gpu=%s --format=csv,noheader' % ','.join(columns)
    smi_output = subprocess.check_output(query, shell=True).decode().strip()

    gpustats = []
    for line in smi_output.split('\n'):
        if not line:
            continue
        gpustat = line.split(',')
        gpustats.append({k: v.strip() for k, v in zip(columns, gpustat)})

    return gpustats


def assign_gpus(num_gpu, memory_threshold=1000):    # (MiB)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    columns = ['index', 'memory.used']
    gpustats = {i['index']: i['memory.used'] for i in check_gpustats(columns)}



    available_gpus = []
    for gpu in sorted(gpustats.keys()):
        if int(gpustats.get(gpu).split(' ')[0]) < memory_threshold:
            available_gpus.append(gpu)

    if len(available_gpus) < num_gpu:
        raise MemoryError('{} GPUs requested, but only {} available'.format(num_gpu, len(available_gpus)))

    gpus_to_assign = available_gpus[:num_gpu]
    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpus_to_assign)
    return gpus_to_assign



def create_args(*args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=args[0])
    parser.add_argument('--hidden_dim', type=int, default=args[1])
    parser.add_argument('--c', type=int, default=args[2])
    parser.add_argument('--num_layers', type=int, default=args[3])
    parser.add_argument('--bias', type=bool, default=args[4])
    parser.add_argument('--act', type=str, default=args[5])
    parser.add_argument('--grad_clip', type=float, default=args[6])
    parser.add_argument('--optimizer', type=str, default=args[7])
    parser.add_argument('--weight_decay', type=float, default=args[8])
    parser.add_argument('--lr', type=float, default=args[9])
    parser.add_argument('--gamma', type=float, default=args[10])
    parser.add_argument('--lr_reduce_freq', type=int, default=args[11])
    parser.add_argument('--cuda', type=int, default=args[12])
    parser.add_argument('--epochs', type=int, default=args[13])
    parser.add_argument('--min_epochs', type=int, default=args[14])
    parser.add_argument('--patience', type=int, default=args[15])
    parser.add_argument('--seed', type=int, default=args[16])
    parser.add_argument('--log_freq', type=int, default=args[17])
    parser.add_argument('--eval_freq', type=int, default=args[18])
    parser.add_argument('--val_prop', type=float, default=args[19])
    parser.add_argument('--test_prop', type=float, default=args[20])
    parser.add_argument('--double_precision', type=int, default=args[21])
    parser.add_argument('--dropout', type=float, default=args[22])
    parser.add_argument('--lambda_rec', type=float, default=args[23])
    parser.add_argument('--lambda_lp', type=float, default=args[24])
    parser.add_argument('--num_dec_layers', type=int, default=args[25])
    parser.add_argument('--use_att', type=bool, default=args[26])
    parser.add_argument('--att_type', type=str, default=args[27])
    parser.add_argument('--att_logit', type=str, default=args[28])
    parser.add_argument('--beta', type=float, default=args[29])
    parser.add_argument('--classifier', type=str, default=args[30])
    parser.add_argument('--clusterer', type=str, default=args[31])
    parser.add_argument('--normalize_adj', type=bool, default=args[32])
    parser.add_argument('--normalize_feats', type=bool, default=args[33])
    parser.add_argument('--anomaly_detector', type=str, default=args[34])
    flags, unknown = parser.parse_known_args()
    return flags



from Ghypeddings.classifiers import *
def get_classifier(args,X,y):
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
    
from Ghypeddings.clusterers import *
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
    

from Ghypeddings.anomaly_detection import *
def get_anomaly_detection_algorithm(algorithm,X,y):
    if(algorithm == 'isolation_forest'):
        return isolation_forest(X,y)
    elif(algorithm == 'one_class_svm'):
        return one_class_svm(X,y)
    elif(algorithm == 'dbscan'):
        return dbscan(X,y)
    elif(algorithm == 'kmeans'):
        return kmeans(X,y)
    elif(algorithm == 'local_outlier_factor'):
        return local_outlier_factor(X,y)
    else:
        raise NotImplementedError