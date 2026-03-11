import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import Ghypeddings.H2HGCN.models.encoders as encoders
from Ghypeddings.H2HGCN.models.encoders import H2HGCN
from Ghypeddings.H2HGCN.models.decoders import model2decoder
from Ghypeddings.H2HGCN.utils.eval_utils import acc_f1
from Ghypeddings.H2HGCN.manifolds import LorentzManifold

 
class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.c = torch.Tensor([1.]).cuda().to(args.device)
        args.manifold = self.manifold = LorentzManifold(args)
        args.feat_dim = args.feat_dim + 1
        # add 1 for Lorentz as the degree of freedom is d - 1 with d dimensions
        args.dim = args.dim + 1
        self.nnodes = args.n_nodes
        self.encoder = H2HGCN(args, 1)

    def encode(self, x, hgnn_adj, hgnn_weight):
        h = self.encoder.encode(x, hgnn_adj, hgnn_weight)
        return h

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError


class NCModel(BaseModel):
    """
    Base model for node classification task.
    """

    def __init__(self, args):
        super(NCModel, self).__init__(args)
        self.decoder = model2decoder(self.c, args)
        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'
        
        self.weights = torch.Tensor([1.] * args.n_classes)
        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)

    def decode(self, h, adj, idx):
        output = self.decoder.decode(h, adj)
        return F.log_softmax(output[idx], dim=1)


    def compute_metrics(self, embeddings, data, split):
        idx = data[f'idx_{split}']
        output = self.decode(embeddings, data['adj_train_norm'], idx)
        loss = F.nll_loss(output, data['labels'][idx], self.weights)
        acc, f1 , recall,precision,roc_auc = acc_f1(output, data['labels'][idx], average=self.f1_average)
        metrics = {'loss': loss, 'acc': acc, 'f1': f1 , 'recall':recall,'precision':precision,'roc_auc':roc_auc}
        return metrics

    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1["f1"] < m2["f1"]