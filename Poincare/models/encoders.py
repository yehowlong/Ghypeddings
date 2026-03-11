"""Graph encoders."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import Ghypeddings.Poincare.manifolds as manifolds

class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x):
        pass

class Shallow(Encoder):
    """
    Shallow Embedding method.
    Learns embeddings or loads pretrained embeddings and uses an MLP for classification.
    """

    def __init__(self, c, args):
        super(Shallow, self).__init__(c)
        self.manifold = getattr(manifolds, 'PoincareBall')()
        weights = torch.Tensor(args.n_nodes, args.dim)
        weights = self.manifold.init_weights(weights, self.c)
        trainable = True
        self.lt = manifolds.ManifoldParameter(weights, trainable, self.manifold, self.c)
        self.all_nodes = torch.LongTensor(list(range(args.n_nodes)))
        layers = []
        self.layers = nn.Sequential(*layers)

    def encode(self, x):
        h = self.lt[self.all_nodes, :]
        h = torch.cat((h, x), 1)
        return h
