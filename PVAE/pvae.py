import sys
sys.path.append(".")
sys.path.append("..")
import os
import datetime
from collections import defaultdict
import torch
from torch import optim
import numpy as np
import logging
import time

from Ghypeddings.PVAE.utils import probe_infnan , process_data , create_args , get_classifier,get_clustering_algorithm,get_anomaly_detection_algorithm
import Ghypeddings.PVAE.objectives as objectives
from Ghypeddings.PVAE.models import Tabular

from Ghypeddings.classifiers import calculate_metrics

runId = datetime.datetime.now().isoformat().replace(':','_')
torch.backends.cudnn.benchmark = True

class PVAE:
    def __init__(self,
                adj,
                features,
                labels,
                dim,
                hidden_dim,
                num_layers=2,
                c=1.0,
                act='relu',
                lr=0.01,
                cuda=0,
                epochs=50,
                seed=42,
                eval_freq=1,
                val_prop=0.,
                test_prop=0.3,
                dropout=0.1,
                beta1=0.9,
                beta2=.999,
                K=1,
                beta=.2,
                analytical_kl=True,
                posterior='WrappedNormal',
                prior='WrappedNormal',
                prior_iso=True,
                prior_std=1.,
                learn_prior_std=True,
                enc='Mob',
                dec='Geo',
                bias=True,
                alpha=0.5,
                classifier=None,
                clusterer=None,
                log_freq=1,
                normalize_adj=False,
                normalize_feats=True,
                anomaly_detector=None
                ):

        self.args = create_args(dim,hidden_dim,num_layers,c,act,lr,cuda,epochs,seed,eval_freq,val_prop,test_prop,dropout,beta1,beta2,K,beta,analytical_kl,posterior,prior,prior_iso,prior_std,learn_prior_std,enc,dec,bias,alpha,classifier,clusterer,log_freq,normalize_adj,normalize_feats,anomaly_detector)
        self.args.n_classes = len(np.unique(labels))
        self.args.feat_dim = features.shape[1]
        self.data = process_data(self.args,adj,features,labels)
        self.args.data_size = [adj.shape[0],self.args.feat_dim]
        self.args.batch_size=1

        self.cls = None

        if int(self.args.cuda) >= 0:
            torch.cuda.manual_seed(self.args.seed)
            self.args.device = 'cuda:' + str(self.args.cuda) if int(self.args.cuda) >= 0 else 'cpu'
        else:
            self.args.device = 'cpu'

        self.args.prior_iso = self.args.prior_iso or self.args.posterior == 'RiemannianNormal'

        # Choosing and saving a random seed for reproducibility
        if self.args.seed == 0: self.args.seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = True
        self.model = Tabular(self.args).to(self.args.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, amsgrad=True, betas=(self.args.beta1, self.args.beta2))
        self.loss_function = getattr(objectives,'vae_objective')

        if self.args.cuda is not None and int(self.args.cuda) >= 0 :
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.args.cuda)
            self.model = self.model.to(self.args.device)
            for x, val in self.data.items():
                if torch.is_tensor(self.data[x]):
                    self.data[x] = self.data[x].to(self.args.device)

        self.tb_embeddings = None


    def fit(self):

        tot_params = sum([np.prod(p.size()) for p in self.model.parameters()])
        logging.info(f"Total number of parameters: {tot_params}")

        t_total = time.time()
        agg = defaultdict(list)
        b_loss, b_recon, b_kl , b_mlik , tb_loss = sys.float_info.max, sys.float_info.max ,sys.float_info.max,sys.float_info.max,sys.float_info.max
        
        best_losses = []
        train_losses = []
        val_losses = []

        for epoch in range(self.args.epochs):
            self.model.train()
            self.optimizer.zero_grad()

            qz_x, px_z, lik, kl, loss , embeddings = self.loss_function(self.model,self.data['idx_train'], self.data['features'], self.data['adj_train'], K=self.args.K, beta=self.args.beta, components=True, analytical_kl=self.args.analytical_kl)
            probe_infnan(loss, "Training loss:")
            loss.backward()
            self.optimizer.step()

            t_loss = loss.item() / len(self.data['idx_train'])
            t_recon = -lik.mean(0).sum().item() / len(self.data['idx_train'])
            t_kl = kl.sum(-1).mean(0).sum().item() / len(self.data['idx_train'])

            if(t_loss < b_loss):
                b_loss = t_loss 
                b_recon = t_recon 
                b_kl = t_kl 


            agg['train_loss'].append(t_loss )
            agg['train_recon'].append(t_recon )
            agg['train_kl'].append(t_kl )

            train_losses.append(t_recon)
            if(len(best_losses) == 0):
                best_losses.append(train_losses[0])
            elif (best_losses[-1] > train_losses[-1]):
                best_losses.append(train_losses[-1])
            else:
                best_losses.append(best_losses[-1])

            if (epoch + 1) % self.args.log_freq == 0:
                print('====> Epoch: {:03d} Loss: {:.2f} Recon: {:.2f} KL: {:.2f}'.format(epoch, agg['train_loss'][-1], agg['train_recon'][-1], agg['train_kl'][-1]))

            if (epoch + 1) % self.args.eval_freq == 0 and self.args.val_prop:
                self.model.eval()
                with torch.no_grad():
                    qz_x, px_z, lik, kl, loss , embeddings= self.loss_function(self.model,self.data['idx_val'], self.data['features'],self.data['adj_train'], K=self.args.K, beta=self.args.beta, components=True)
                    tt_loss = loss.item() / len(self.data['idx_val'])
                    val_losses.append(tt_loss)
                    if(tt_loss < tb_loss):
                        tb_loss = tt_loss 
                        self.tb_embeddings = embeddings[0]

                    agg['test_loss'].append(tt_loss )
                    print('====>             Test loss: {:.4f}'.format(agg['test_loss'][-1]))


        logging.info("Optimization Finished!")
        logging.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        print('====> Training: Best Loss: {:.2f} Best Recon: {:.2f} Best KL: {:.2f}'.format(b_loss,b_recon,b_kl))
        print('====> Testing: Best Loss: {:.2f}'.format(tb_loss))

        train_idx = self.data['idx_train']
        val_idx = self.data['idx_val']
        idx = np.unique(np.concatenate((train_idx,val_idx)))
        X =  self.model.manifold.logmap0(self.tb_embeddings[idx]).cpu().detach().numpy()
        y = self.data['labels'].cpu().reshape(-1,1)[idx]

        if(self.args.classifier):
            self.cls = get_classifier(self.args, X,y)
            acc,f1,recall,precision,roc_auc = calculate_metrics(self.cls,X,y)
        elif self.args.clusterer:
            y = y.reshape(-1,)
            acc,f1,recall,precision,roc_auc = get_clustering_algorithm(self.args.clusterer,X,y)[6:]
        elif self.args.anomaly_detector:
            y = y.reshape(-1,)
            acc,f1,recall,precision,roc_auc = get_anomaly_detection_algorithm(self.args.anomaly_detector,X,y)[6:]

        return {'train':train_losses,'best':best_losses,'val':val_losses},acc,f1,recall,precision,roc_auc,time.time() - t_total

    def predict(self):
        self.model.eval()
        with torch.no_grad():
            qz_x, px_z, lik, kl, loss , embeddings=self.loss_function(self.model,self.data['idx_test'], self.data['features'],self.data['adj_train'], K=self.args.K, beta=self.args.beta, components=True)
            tt_loss = loss.item() / len(self.data['idx_test'])
        test_idx = self.data['idx_test']
        data = self.model.manifold.logmap0(embeddings[0][test_idx]).cpu().detach().numpy()
        labels = self.data['labels'].reshape(-1,1).cpu()[test_idx]
        if self.args.classifier:
            acc,f1,recall,precision,roc_auc = calculate_metrics(self.cls,data,labels)
        elif self.args.clusterer:
            labels = labels.reshape(-1,)
            acc,f1,recall,precision,roc_auc = get_clustering_algorithm(self.args.clusterer,data,labels)[6:]
        elif self.args.anomaly_detector:
            labels = labels.reshape(-1,)
            acc,f1,recall,precision,roc_auc = get_anomaly_detection_algorithm(self.args.anomaly_detector,data,labels)[6:]
        self.tb_embeddings = embeddings[0]
        return abs(tt_loss) , acc, f1 , recall,precision,roc_auc


    def save_embeddings(self,directory):
        tb_embeddings_euc = self.model.manifold.logmap0(self.tb_embeddings)
        for_classification_hyp = np.hstack((self.tb_embeddings.cpu().detach().numpy(),self.data['labels'].reshape(-1,1).cpu()))
        for_classification_euc = np.hstack((tb_embeddings_euc.cpu().detach().numpy(),self.data['labels'].reshape(-1,1).cpu()))
        hyp_file_path = os.path.join(directory,'pvae_embeddings_hyp.csv')
        euc_file_path = os.path.join(directory,'pvae_embeddings_euc.csv')
        np.savetxt(hyp_file_path, for_classification_hyp, delimiter=',')
        np.savetxt(euc_file_path, for_classification_euc, delimiter=',')
