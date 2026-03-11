"""Graph decoders."""
import Ghypeddings.HGCAE.manifolds as manifolds
import torch.nn as nn
import torch.nn.functional as F


import torch


class Decoder(nn.Module):
    """
    Decoder abstract class
    """

    def __init__(self, c):
        super(Decoder, self).__init__()
        self.c = c

    def classify(self, x, adj):
        '''
        output
        - nc : probs 
        - rec : input_feat
        '''
        if self.decode_adj:
            input = (x, adj)
            output, _ = self.classifier.forward(input)
        else:
            output = self.classifier.forward(x)
        return output


    def decode(self, x, adj):
        '''
        output
        - nc : probs 
        - rec : input_feat
        '''
        if self.decode_adj:
            input = (x, adj)
            output, _ = self.decoder.forward(input)
        else:
            output = self.decoder.forward(x)
        return output



import Ghypeddings.HGCAE.layers.hyp_layers as hyp_layers
class HGCAEDecoder(Decoder):
    """
    Decoder for HGCAE
    """

    def __init__(self, c, args, task):
        super(HGCAEDecoder, self).__init__(c)
        self.manifold = getattr(manifolds, 'PoincareBall')()
    
        assert args.num_layers > 0

        dims, acts, _ = hyp_layers.get_dim_act_curv(args)
        dims = dims[::-1]
        acts = acts[::-1][:-1] + [lambda x: x] # Last layer without act
        self.curvatures = self.c[::-1]

        encdec_share_curvature = False
        if not encdec_share_curvature and args.num_layers == args.num_dec_layers: # do not share and enc-dec mirror-shape
            num_c = len(self.curvatures)
            self.curvatures = self.curvatures[:1] 
            if args.c_trainable == 1:
                self.curvatures += [nn.Parameter(torch.Tensor([args.c]).to(args.device))] * (num_c - 1)
            else:
                self.curvatures += [torch.tensor([args.c])] * (num_c - 1)
                if not args.cuda == -1:
                    self.curvatures = [curv.to(args.device) for curv in self.curvatures]


        self.curvatures = self.curvatures[:-1] + [None]


        hgc_layers = []
        num_dec_layers = args.num_dec_layers
        for i in range(num_dec_layers):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                hyp_layers.HyperbolicGraphConvolution(
                        self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att,
                        att_type=args.att_type, att_logit=args.att_logit, beta=args.beta, decode=True
                )
            )

        self.decoder = nn.Sequential(*hgc_layers)
        self.decode_adj = True

    # NOTE : self.c is fixed, not trainable
    def classify(self, x, adj):
        h = self.manifold.logmap0(x, c=self.c)
        return super(HGCAEDecoder, self).classify(h, adj)
    
    def decode(self, x, adj):
        output = super(HGCAEDecoder, self).decode(x, adj)
        return output

model2decoder = HGCAEDecoder

