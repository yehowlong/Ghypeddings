"""Graph encoders."""
import HGCAE.manifolds as manifolds
import HGCAE.layers.hyp_layers as hyp_layers
import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, c, use_cnn=None):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x, adj):
        self.features = []
        if self.encode_graph:
            input = (x, adj)
            xx = input
            for i in range(len(self.layers)):
                out = self.layers[i].forward(xx)
                self.features.append(out[0])
                xx = out
            output , _ = xx
        else:
            output = self.layers.forward(x)
        return output

class HGCAE(Encoder):
    """
    Hyperbolic Graph Convolutional Auto-Encoders.
    """

    def __init__(self, c, args): #, use_cnn
        super(HGCAE, self).__init__(c, use_cnn=True)
        self.manifold = getattr(manifolds, "PoincareBall")()
        assert args.num_layers > 0 
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        if args.c_trainable == 1: 
            self.curvatures.append(nn.Parameter(torch.Tensor([args.c]).to(args.device)))
        else:
            self.curvatures.append(torch.tensor([args.c]).to(args.device)) 
        hgc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]

            hgc_layers.append(
                    hyp_layers.HyperbolicGraphConvolution(
                            self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att,
                            att_type=args.att_type, att_logit=args.att_logit, beta=args.beta
                    )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def encode(self, x, adj):
        self.curvatures[0] = torch.clamp_min(self.curvatures[0],min=1e-12)
        x_hyp = self.manifold.proj(
                self.manifold.expmap0(self.manifold.proj_tan0(x, self.curvatures[0]), c=self.curvatures[0]),
                c=self.curvatures[0])
        return super(HGCAE, self).encode(x_hyp, adj)
