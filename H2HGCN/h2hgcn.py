from __future__ import division
from __future__ import print_function
import logging
import os
import time
import numpy as np
import torch
from Ghypeddings.H2HGCN.models.base_models import NCModel
from Ghypeddings.H2HGCN.utils.data_utils import process_data
from Ghypeddings.H2HGCN.utils.train_utils import format_metrics, create_args
from Ghypeddings.H2HGCN.utils.pre_utils import *
import warnings
warnings.filterwarnings('ignore')

class H2HGCN:
    def __init__(self,
                adj,
                features,
                labels,
                dim,
                c=None,
                num_layers=2,
                bias=True,
                act='leaky_relu',
                select_manifold='lorentz',
                num_centroid=10,
                lr_stie=0.009,
                stie_vars=[],
                stiefel_optimizer='rsgd',
                eucl_vars=[],
                grad_clip=None,
                optimizer='Adam',
                weight_decay=0.01,
                lr=0.01,
                lr_scheduler='step',
                lr_gamma=0.5,
                step_lr_gamma=0.1,
                step_lr_reduce_freq=500,
                proj_init='xavier',
                tie_weight=True,
                cuda=0,
                epochs=50,
                min_epochs=50,
                patience=None,
                seed=42,
                log_freq=1,
                eval_freq=1,
                val_prop=0.15,
                test_prop=0.15,
                double_precision=0,
                dropout=0.1,
                normalize_adj=False,
                normalize_feats=True
                ):
        
        self.args = create_args(dim,c,num_layers,bias,act,select_manifold,num_centroid,lr_stie,stie_vars,stiefel_optimizer,eucl_vars,grad_clip,optimizer,weight_decay,lr,lr_scheduler,lr_gamma,step_lr_gamma,step_lr_reduce_freq,proj_init,tie_weight,cuda,epochs,min_epochs,patience,seed,log_freq,eval_freq,val_prop,test_prop,double_precision,dropout,normalize_adj,normalize_feats)
        
        self.args.n_nodes = adj.shape[0]
        self.args.feat_dim = features.shape[1]
        self.args.n_classes = len(np.unique(labels))
        self.data = process_data(self.args,adj,features,labels)

        if int(self.args.double_precision):
            torch.set_default_dtype(torch.float64)

        self.args.device = 'cuda:' + str(self.args.cuda) if int(self.args.cuda) >= 0 else 'cpu'
        self.args.patience = self.args.epochs if not self.args.patience else  int(self.args.patience)
        self.model = NCModel(self.args)
        self.optimizer, self.lr_scheduler, self.stiefel_optimizer, self.stiefel_lr_scheduler = set_up_optimizer_scheduler(True, self.args, self.model, self.args.lr, self.args.lr_stie)
        
        if self.args.cuda is not None and int(self.args.cuda) >= 0 :
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.args.cuda)
            self.model = self.model.to(self.args.device)
            for x, val in self.data.items():
                if torch.is_tensor(self.data[x]):
                    self.data[x] = self.data[x].to(self.args.device)
        self.best_emb = None


    def fit(self):
        logging.getLogger().setLevel(logging.INFO)
        logging.info(f'Using: {self.args.device}')
        tot_params = sum([np.prod(p.size()) for p in self.model.parameters()])
        logging.info(f"Total number of parameters: {tot_params}")
        
        t_total = time.time()
        counter = 0
        best_val_metrics = self.model.init_metric_dict()

        best_losses = []
        real_losses = []
        train_losses = []

        for epoch in range(self.args.epochs):       
            t = time.time()
            self.model.train()
            self.optimizer.zero_grad()
            self.stiefel_optimizer.zero_grad()
            embeddings = self.model.encode(self.data['features'], self.data['hgnn_adj'], self.data['hgnn_weight']) 
            train_metrics = self.model.compute_metrics(embeddings, self.data, 'train')
            train_metrics['loss'].backward()
            if self.args.grad_clip is not None:
                max_norm = float(self.args.grad_clip)
                all_params = list(self.model.parameters())
                for param in all_params:
                    torch.nn.utils.clip_grad_norm_(param, max_norm)
            self.optimizer.step()
            self.stiefel_optimizer.step()
            self.lr_scheduler.step()
            self.stiefel_lr_scheduler.step()

            train_losses.append(train_metrics['loss'].item())
            if(len(best_losses) == 0):
                best_losses.append(train_losses[0])
            elif (best_losses[-1] > train_losses[-1]):
                best_losses.append(train_losses[-1])
            else:
                best_losses.append(best_losses[-1])

            if (epoch + 1) % self.args.log_freq == 0:
                logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1),
                                    'lr: {:04f}, stie_lr: {:04f}'.format(self.lr_scheduler.get_lr()[0], self.stiefel_lr_scheduler.get_lr()[0]),
                                    format_metrics(train_metrics, 'train'),
                                    'time: {:.4f}s'.format(time.time() - t)
                                    ]))
                
            if (epoch + 1) % self.args.eval_freq == 0:
                self.model.eval()
                embeddings = self.model.encode(self.data['features'], self.data['hgnn_adj'], self.data['hgnn_weight'])
                val_metrics = self.model.compute_metrics(embeddings, self.data, 'val')
                real_losses.append(val_metrics['loss'].item())
                if (epoch + 1) % self.args.log_freq == 0:
                    logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(val_metrics, 'val')]))
                if self.model.has_improved(best_val_metrics, val_metrics):
                    self.best_emb = embeddings
                    best_val_metrics = val_metrics
                    counter = 0
                else:
                    counter += 1
                    if counter == self.args.patience and epoch > self.args.min_epochs:
                        logging.info("Early stopping")
                        break

        logging.info("Training Finished!")
        logging.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        return {'val':real_losses,'best':best_losses,'train':train_losses},best_val_metrics['acc'],best_val_metrics['f1'],best_val_metrics['recall'],best_val_metrics['precision'],best_val_metrics['roc_auc'],time.time() - t_total

    def predict(self):
        self.model.eval()
        embeddings = self.model.encode(self.data['features'], self.data['hgnn_adj'], self.data['hgnn_weight'])
        val_metrics = self.model.compute_metrics(embeddings, self.data, 'test')
        logging.info(" ".join([format_metrics(val_metrics, 'test')]))
        return val_metrics['loss'].item(),val_metrics['acc'],val_metrics['f1'],val_metrics['recall'],val_metrics['precision'],val_metrics['roc_auc']
    
    def save_embeddings(self):
        #tb_embeddings_euc = self.model.manifold.log_map_zero(self.best_emb)
        for_classification_hyp = np.hstack((self.best_emb.cpu().detach().numpy(),self.data['labels'].cpu().reshape(-1,1)))
        #for_classification_euc = np.hstack((tb_embeddings_euc.cpu().detach().numpy(),self.data['labels'].cpu().reshape(-1,1)))
        hyp_file_path = os.path.join(os.getcwd(),'h2hgcn_embeddings_hyp.csv')
        #euc_file_path = os.path.join(os.getcwd(),'h2hgcn_embeddings_euc.csv')
        np.savetxt(hyp_file_path, for_classification_hyp, delimiter=',')
        #np.savetxt(euc_file_path, for_classification_euc, delimiter=',')
