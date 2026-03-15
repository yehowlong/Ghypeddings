import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import prod
from PVAE.utils import Constants
from PVAE.ops.manifold_layers import GeodesicLayer, MobiusLayer, LogZero, ExpZero
from torch.nn.modules.module import Module

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
    dims = [args.feat_dim] + ([args.hidden_dim] * (args.num_layers - 1))

    return dims, acts


class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x, adj):
        input = (x, adj)
        output, _ = self.layers.forward(input)
        return output

class GraphConvolution(Module):
    """
    Simple GCN layer.
    """

    def __init__(self, in_features, out_features, dropout, act, use_bias):
        super(GraphConvolution, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.act = act
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, input):
        x, adj = input
        hidden = self.linear.forward(x)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        if adj.is_sparse:
            support = torch.spmm(adj, hidden)
        else:
            support = torch.mm(adj, hidden)
        output = self.act(support), adj
        return output

    def extra_repr(self):
        return 'input_dim={}, output_dim={}'.format(
                self.in_features, self.out_features
        )

class GCN(Encoder):
    """
    Graph Convolution Networks.
    """

    def __init__(self, c, args):
        super(GCN, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        gc_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            gc_layers.append(GraphConvolution(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*gc_layers)


def extra_hidden_layer(hidden_dim, non_lin):
     return nn.Sequential(nn.Linear(hidden_dim, hidden_dim), non_lin)       

class EncWrapped(nn.Module):
    """ Usual encoder followed by an exponential map """
    def __init__(self,c,args, manifold, data_size, non_lin, num_hidden_layers, hidden_dim, prior_iso):
        super(EncWrapped, self).__init__()
        self.manifold = manifold
        self.data_size = data_size
        self.enc = GCN(c,args)
        self.fc21 = nn.Linear(hidden_dim, manifold.coord_dim)
        self.fc22 = nn.Linear(hidden_dim, manifold.coord_dim if not prior_iso else 1)

    def forward(self,adj,x):
        e = self.enc.encode(x,adj)
        mu = self.fc21(e)          # flatten data
        mu = self.manifold.expmap0(mu)
        return mu, F.softplus(self.fc22(e)) + Constants.eta,  self.manifold


class DecWrapped(nn.Module):
    """ Usual encoder preceded by a logarithm map """
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim):
        super(DecWrapped, self).__init__()
        self.data_size = data_size
        self.manifold = manifold
        modules = []
        modules.append(nn.Sequential(nn.Linear(manifold.coord_dim, hidden_dim), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.dec = nn.Sequential(*modules)
        # self.fc31 = nn.Linear(hidden_dim, prod(data_size))
        self.fc31 = nn.Linear(hidden_dim, data_size[1])

    def forward(self, z):
        z = self.manifold.logmap0(z)
        d = self.dec(z)
        # mu = self.fc31(d).view(*z.size()[:-1], *self.data_size)  # reshape data
        mu = self.fc31(d).view(*z.size()[:-1], 1, self.data_size[1]) 
        return mu, torch.ones_like(mu)


class DecGeo(nn.Module):
    """ First layer is a Hypergyroplane followed by usual decoder """
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim):
        super(DecGeo, self).__init__()
        self.data_size = data_size
        modules = []
        modules.append(nn.Sequential(GeodesicLayer(manifold.coord_dim, hidden_dim, manifold), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc31 = nn.Linear(hidden_dim, data_size[1])

    def forward(self, z):
        d = self.dec(z)
        # mu = self.fc31(d).view(*z.size()[:-1], *self.data_size)  # reshape data
        mu = self.fc31(d).view(*z.size()[:-1], 1, self.data_size[1]) 
        return mu, torch.ones_like(mu)


class EncMob(nn.Module):
    """ Last layer is a Mobius layers """
    def __init__(self,c,args, manifold, data_size, non_lin, num_hidden_layers, hidden_dim, prior_iso):
        super(EncMob, self).__init__()
        self.manifold = manifold
        self.data_size = data_size
        # modules = []
        # modules.append(nn.Sequential(nn.Linear(data_size[1], hidden_dim), non_lin))
        # modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        # self.enc = nn.Sequential(*modules)
        self.enc = GCN(c,args)
        self.fc21 = MobiusLayer(hidden_dim, manifold.coord_dim, manifold)
        self.fc22 = nn.Linear(hidden_dim, manifold.coord_dim if not prior_iso else 1)

    def forward(self,adj,x):
        #e = self.enc(x.view(*x.size()[:-len(self.data_size)], -1))            # flatten data
        e = self.enc.encode(x,adj)
        mu = self.fc21(e)          # flatten data
        mu = self.manifold.expmap0(mu)
        return mu, F.softplus(self.fc22(e)) + Constants.eta,  self.manifold


class DecMob(nn.Module):
    """ First layer is a Mobius Matrix multiplication """
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim):
        super(DecMob, self).__init__()
        self.data_size = data_size
        modules = []
        modules.append(nn.Sequential(MobiusLayer(manifold.coord_dim, hidden_dim, manifold), LogZero(manifold), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc31 = nn.Linear(hidden_dim, prod(data_size))

    def forward(self, z):
        d = self.dec(z)
        mu = self.fc31(d).view(*z.size()[:-1], *self.data_size)  # reshape data
        return mu, torch.ones_like(mu)
