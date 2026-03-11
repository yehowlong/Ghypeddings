"""Euclidean layers."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


def get_dim_act(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers - 1)

    dims = [args.feat_dim]
    if args.num_layers > 1:
        # Check layer_num and hdden_dim match
        hidden_dim = [int(h) for h in args.hidden_dim.split(',')]
        if args.num_layers != len(hidden_dim) + 1:
            raise RuntimeError('Check dimension hidden:{}, num_laysers:{}'.format(args.hidden_dim, args.num_layers) )
        dims = dims + hidden_dim

    dims += [args.dim]
    acts += [act]
    return dims, acts


class Linear(Module):
    """
    Simple Linear layer with dropout.
    """

    def __init__(self, in_features, out_features, dropout, act, use_bias):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.act = act

    def forward(self, x):
        hidden = self.linear.forward(x)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        out = self.act(hidden)
        return out

'''
InnerProductDecdoer implemntation from:
https://github.com/zfjsail/gae-pytorch/blob/master/gae/model.py
'''
class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout=0, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, emb_in, emb_out):
        cos_dist = emb_in * emb_out
        probs = self.act(cos_dist.sum(1))
        return probs
