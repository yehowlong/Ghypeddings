import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.modules.loss
import argparse

def format_metrics(metrics, split):
    """Format metric in metric dict for logging."""
    return " ".join(
            ["{}_{}: {:.4f}".format(split, metric_name, metric_val) for metric_name, metric_val in metrics.items()])

def create_args(*args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=args[0])
    parser.add_argument('--c', type=int, default=args[1])
    parser.add_argument('--num_layers', type=int, default=args[2])
    parser.add_argument('--bias', type=bool, default=args[3])
    parser.add_argument('--act', type=str, default=args[4])
    parser.add_argument('--select_manifold', type=str, default=args[5])
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
    parser.add_argument('--use_att', type=bool, default=args[23])
    parser.add_argument('--alpha', type=float, default=args[24])
    parser.add_argument('--local_agg', type=bool, default=args[25])
    parser.add_argument('--normalize_adj', type=bool, default=args[26])
    parser.add_argument('--normalize_feats', type=bool, default=args[27])
    parser.add_argument('--model', type=str, default='GAT') #GCN, GAT,HGCN
    parser.add_argument('--n_heads', type=int, default=1) #GCN, GAT,HGCN
    flags, unknown = parser.parse_known_args()
    return flags