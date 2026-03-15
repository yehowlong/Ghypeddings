from HGCAE.models.base_models import LPModel
import logging
import torch
import numpy as np
import os
import time
from HGCAE.utils.train_utils import get_dir_name, format_metrics
from HGCAE.utils.data_utils import process_data
from HGCAE.utils.train_utils import create_args , get_classifier ,get_clustering_algorithm,get_anomaly_detection_algorithm
import HGCAE.optimizers as optimizers
from HGCAE.utils.data_utils import sparse_mx_to_torch_sparse_tensor

from classifiers import calculate_metrics

class HGCAE(object):
    def __init__(self, 
                adj,
                features,
                labels,
                dim,
                hidden_dim,
                c=None,
                num_layers=2,
                bias=True,
                act='relu',
                grad_clip=None,
                optimizer='RiemannianAdam',
                weight_decay=0.01,
                lr=0.001,
                gamma=0.5,
                lr_reduce_freq=500,
                cuda=0,
                epochs=50,
                min_epochs=50,
                patience=None,
                seed=42,
                log_freq=1,
                eval_freq=1,
                val_prop=0.0002,
                test_prop=0.3,
                double_precision=0,
                dropout=0.1,
                lambda_rec=1.0,
                lambda_lp=1.0,
                num_dec_layers=2,
                use_att= True,
                att_type= 'sparse_adjmask_dist',
                att_logit='tanh',
                beta = 0.2,
                classifier=None,
                clusterer = None,
                normalize_adj=False,
                normalize_feats=True,
                anomaly_detector=None
                ):
        
        self.args = create_args(dim,hidden_dim,c,num_layers,bias,act,grad_clip,optimizer,weight_decay,lr,gamma,lr_reduce_freq,cuda,epochs,min_epochs,patience,seed,log_freq,eval_freq,val_prop,test_prop,double_precision,dropout,lambda_rec,lambda_lp,num_dec_layers,use_att,att_type,att_logit,beta,classifier,clusterer,normalize_adj,normalize_feats,anomaly_detector)
        self.cls = None

        self.args.n_nodes = adj.shape[0]
        self.args.feat_dim = features.shape[1]
        self.args.n_classes = len(np.unique(labels))
        self.data = process_data(self.args,adj,features,labels)

        if(self.args.c == None):
            self.args.c_trainable = 1
            self.args.c = 1.0
        
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        if int(self.args.double_precision):
            torch.set_default_dtype(torch.float64)
        if int(self.args.cuda) >= 0:
            torch.cuda.manual_seed(self.args.seed)

        self.args.device = 'cuda:' + str(self.args.cuda) if int(self.args.cuda) >= 0 else 'cpu'
        self.args.patience = self.args.epochs if not self.args.patience else  int(self.args.patience)

        if not self.args.lr_reduce_freq:
            self.args.lr_reduce_freq = self.args.epochs

        self.args.nb_false_edges = len(self.data['train_edges_false'])
        self.args.nb_edges = len(self.data['train_edges'])
        st0 = np.random.get_state()
        self.args.np_seed = st0
        np.random.set_state(self.args.np_seed)

        for x, val in self.data.items():
            if 'adj' in x:
                self.data[x] = sparse_mx_to_torch_sparse_tensor(self.data[x])

        self.model = LPModel(self.args)

        if self.args.cuda is not None and int(self.args.cuda) >= 0 :
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.args.cuda)
            self.model = self.model.to(self.args.device)
            for x, val in self.data.items():
                if torch.is_tensor(self.data[x]):
                    self.data[x] = self.data[x].to(self.args.device)

        self.adj_train_enc = self.data['adj_train_enc']
        self.optimizer = getattr(optimizers, self.args.optimizer)(params=self.model.parameters(), lr=self.args.lr,
                                                        weight_decay=self.args.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=int(self.args.lr_reduce_freq),
            gamma=float(self.args.gamma)
        )

        self.best_emb = None



    def fit(self):

        logging.getLogger().setLevel(logging.INFO)
        logging.info(f'Using: {self.args.device}')
        logging.info(str(self.model))
        tot_params = sum([np.prod(p.size()) for p in self.model.parameters()])
        logging.info(f"Total number of parameters: {tot_params}")

        t_total = time.time()
        counter = 0
        best_val_metrics = self.model.init_metric_dict()

        best_losses = []
        train_losses = []
        val_losses = []

        for epoch in range(self.args.epochs):
            t = time.time()
            self.model.train()
            self.optimizer.zero_grad()
            embeddings = self.model.encode(self.data['features'], self.adj_train_enc)
            train_metrics = self.model.compute_metrics(embeddings, self.data, 'train', epoch)
            print(train_metrics)
            train_metrics['loss'].backward()
            if self.args.grad_clip is not None:
                max_norm = float(self.args.grad_clip)
                all_params = list(self.model.parameters())
                for param in all_params:
                    torch.nn.utils.clip_grad_norm_(param, max_norm)
            self.optimizer.step()
            self.lr_scheduler.step()

            train_losses.append(train_metrics['loss'].item())
            if(len(best_losses) == 0):
                best_losses.append(train_losses[0])
            elif (best_losses[-1] > train_losses[-1]):
                best_losses.append(train_losses[-1])
            else:
                best_losses.append(best_losses[-1])

            with torch.no_grad():
                if (epoch + 1) % self.args.log_freq == 0:
                    logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1),
                                           'lr: {}'.format(self.lr_scheduler.get_lr()[0]),
                                           format_metrics(train_metrics, 'train'),
                                           'time: {:.4f}s'.format(time.time() - t)
                                           ]))
                    
                if (epoch + 1) % self.args.eval_freq == 0:
                    self.model.eval()
                    embeddings = self.model.encode(self.data['features'], self.adj_train_enc)
                    #val_metrics = self.model.compute_metrics(embeddings, self.data, 'val')
                    # val_losses.append(val_metrics['loss'].item())
                    # if (epoch + 1) % self.args.log_freq == 0:
                    #     logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(val_metrics, 'val')]))
                    # if self.model.has_improved(best_val_metrics, val_metrics):
                    #     self.best_emb = embeddings
                    #     best_val_metrics = val_metrics
                    #     counter = 0
                    # else:
                    #     counter += 1
                    #     if counter == self.args.patience and epoch > self.args.min_epochs:
                    #         logging.info("Early stopping")
                    #         break

        logging.info("Training Finished!")
        logging.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # train_idx = np.unique(self.data['train_edges'][:,0].cpu().detach().numpy())
        # val_idx = np.unique(self.data['val_edges'][:,0].cpu().detach().numpy())
        # idx = np.unique(np.concatenate((train_idx,val_idx)))
        # X = self.model.manifold.logmap0(self.best_emb[idx],self.model.encoder.curvatures[-1]).cpu().detach().numpy()
        # y = self.data['labels'].reshape(-1,1)[idx]

        # if(self.args.classifier):
        #     self.cls = get_classifier(self.args, X,y)
        #     acc,f1,recall,precision,roc_auc = calculate_metrics(self.cls,X,y)
        # elif self.args.clusterer:
        #     y = y.reshape(-1,)
        #     acc,f1,recall,precision,roc_auc = get_clustering_algorithm(self.args.clusterer,X,y)[6:]
        # elif self.args.anomaly_detector:
        #     y = y.reshape(-1,)
        #     acc,f1,recall,precision,roc_auc = get_anomaly_detection_algorithm(self.args.anomaly_detector,X,y)[6:]
        
        # return {'train':train_losses,'best':best_losses,'val':val_losses},acc,f1,recall,precision,roc_auc , time.time() - t_total
        return {'train':train_losses,'best':best_losses,'val':val_losses}, time.time() - t_total

    def predict(self):
        self.model.eval()
        test_idx = np.unique(self.data['test_edges'][:,0].cpu().detach().numpy())
        embeddings = self.model.encode(self.data['features'], self.adj_train_enc)
        val_metrics = self.model.compute_metrics(embeddings, self.data, 'test')
        data = self.model.manifold.logmap0(embeddings[test_idx],self.model.encoder.curvatures[-1]).cpu().detach().numpy()
        labels = self.data['labels'].reshape(-1,1)[test_idx]
        if self.args.classifier:
            acc,f1,recall,precision,roc_auc = calculate_metrics(self.cls,data,labels)
        elif self.args.clusterer:
            labels = labels.reshape(-1,)
            acc,f1,recall,precision,roc_auc = get_clustering_algorithm(self.args.clusterer,data,labels)[6:]
        elif self.args.anomaly_detector:
            labels = labels.reshape(-1,)
            acc,f1,recall,precision,roc_auc = get_anomaly_detection_algorithm(self.args.anomaly_detector,data,labels)[6:]
        self.tb_embeddings = embeddings
        return val_metrics['loss'].item(),acc,f1,recall,precision,roc_auc

                    
    def save_embeddings(self,directory):
        self.model.eval()
        embeddings = self.model.encode(self.data['features'], self.adj_train_enc)
        tb_embeddings_euc = self.model.manifold.logmap0(embeddings,self.model.encoder.curvatures[-1])
        for_classification_hyp = np.hstack((embeddings.cpu().detach().numpy(),self.data['labels'].reshape(-1,1)))
        for_classification_euc = np.hstack((tb_embeddings_euc.cpu().detach().numpy(),self.data['labels'].reshape(-1,1)))
        hyp_file_path = os.path.join(directory,'hgcae_embeddings_hyp.csv')
        euc_file_path = os.path.join(directory,'hgcae_embeddings_euc.csv')
        np.savetxt(hyp_file_path, for_classification_hyp, delimiter=',')
        np.savetxt(euc_file_path, for_classification_euc, delimiter=',')