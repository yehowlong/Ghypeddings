"""Graph decoders."""
import Ghypeddings.Poincare.manifolds as manifolds
import torch.nn as nn
import torch.nn.functional as F
from Ghypeddings.Poincare.layers.layers import  Linear
import torch

class Decoder(nn.Module):
    """
    Decoder abstract class for node classification tasks.
    """

    def __init__(self, c):
        super(Decoder, self).__init__()
        self.c = c

    def decode(self, x):
        probs = self.cls.forward(x)
        return probs


class LinearDecoder(Decoder):
    """
    MLP Decoder for Hyperbolic/Euclidean node classification models.
    """

    def __init__(self, c, args):
        super(LinearDecoder, self).__init__(c)
        self.manifold = getattr(manifolds, 'PoincareBall')()
        self.input_dim = args.dim + args.feat_dim
        self.output_dim = args.n_classes
        self.bias = True
        self.cls = Linear(self.input_dim, self.output_dim, args.dropout, lambda x: x, self.bias)

    def decode(self, x):
        h = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.c), c=self.c)
        return super(LinearDecoder, self).decode(h)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
                self.input_dim, self.output_dim, self.bias, self.c
        )


model2decoder = LinearDecoder

