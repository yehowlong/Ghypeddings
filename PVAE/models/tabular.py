import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.data import DataLoader

import math
from PVAE.models.vae import VAE

from PVAE.distributions import RiemannianNormal, WrappedNormal
from torch.distributions import Normal
import PVAE.manifolds as manifolds
from PVAE.models.architectures import EncWrapped, DecWrapped, EncMob, DecMob, DecGeo
from PVAE.utils import get_activation

class Tabular(VAE):
    """ Derive a specific sub-class of a VAE for tabular data. """
    def __init__(self, params):
        c = nn.Parameter(params.c * torch.ones(1), requires_grad=False)
        manifold = getattr(manifolds, 'PoincareBall')(params.dim, c)
        super(Tabular, self).__init__(
            eval(params.prior),           # prior distribution
            eval(params.posterior),       # posterior distribution
            dist.Normal,                  # likelihood distribution
            eval('Enc' + params.enc)(params.c,params,manifold, params.data_size, get_activation(params), params.num_layers, params.hidden_dim, params.prior_iso),
            eval('Dec' + params.dec)(manifold, params.data_size, get_activation(params), params.num_layers, params.hidden_dim),
            params
        )
        self.manifold = manifold
        self._pz_mu = nn.Parameter(torch.zeros(1, params.dim), requires_grad=False)
        self._pz_logvar = nn.Parameter(torch.zeros(1, 1), requires_grad=params.learn_prior_std)
        self.modelName = 'Tabular'

    @property
    def pz_params(self):
        return self._pz_mu.mul(1), F.softplus(self._pz_logvar).div(math.log(2)).mul(self.prior_std), self.manifold