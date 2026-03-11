from Ghypeddings.HGNN.task import *
from Ghypeddings.HGNN.utils import *
from Ghypeddings.HGNN.manifold import *
from Ghypeddings.HGNN.gnn import RiemannianGNN

class HGNN:
    def __init__(self,
                adj,
                features,
                labels,
                dim,
                c=None,
                num_layers=2,
                bias=True,
                act='leaky_relu',
                alpha=0.2,
                select_manifold='poincare',
                num_centroid=100,
                eucl_vars=[],
                hyp_vars=[],
                grad_clip=1.0,
                optimizer='sgd',
                weight_decay=0.01,
                lr=0.01,
                lr_scheduler='cosine',
                lr_gamma=0.5,
                lr_hyperbolic=0.01,
                hyper_optimizer='ramsgrad',
                proj_init='xavier',
                tie_weight=True,
                epochs=50,
                patience=100,
                seed=42,
                log_freq=1,
                eval_freq=1,
                val_prop=0.15,
                test_prop=0.15,
                double_precision=0,
                dropout=0.01,
                normalize_adj=False,
                normalize_feats=True):
        
        self.args = create_args(dim,c,num_layers,bias,act,alpha,select_manifold,num_centroid,eucl_vars,hyp_vars,grad_clip,optimizer,weight_decay,lr,lr_scheduler,lr_gamma,lr_hyperbolic,hyper_optimizer,proj_init,tie_weight,epochs,patience,seed,log_freq,eval_freq,val_prop,test_prop,double_precision,dropout,normalize_adj,normalize_feats)
        
        set_seed(self.args.seed)
        self.logger = create_logger()
        if self.args.select_manifold == 'lorentz':
                    self.args.dim += 1
        if self.args.select_manifold == 'lorentz':
            self.manifold= LorentzManifold(self.args, self.logger)
        elif self.args.select_manifold == 'poincare':
            self.manifold= PoincareManifold(self.args,self.logger)
        rgnn = RiemannianGNN(self.args, self.logger, self.manifold)
        self.gnn = NodeClassificationTask(self.args, self.logger, rgnn, self.manifold, adj,features,labels)
    
    def fit(self):
        return self.gnn.run_gnn()

    def predict(self):
        return self.gnn.evaluate(self.gnn.loader, 'test', self.gnn.model, self.gnn.loss_function)

    def save_embeddings(self):
        labels = np.argmax(th.squeeze(self.gnn.labels).numpy(),axis=1)
        #tb_embeddings_euc = self.gnn.manifold.log_map_zero(self.gnn.early_stop.best_emb)
        for_classification_hyp = np.hstack((self.gnn.early_stop.best_emb.cpu().detach().numpy(),labels.reshape(-1,1)))
        #for_classification_euc = np.hstack((tb_embeddings_euc.cpu().detach().numpy(),labels.reshape(-1,1)))
        hyp_file_path = os.path.join(os.getcwd(),'hgnn_embeddings_hyp.csv')
        #euc_file_path = os.path.join(os.getcwd(),'hgnn_embeddings_euc.csv')
        np.savetxt(hyp_file_path, for_classification_hyp, delimiter=',')
        #np.savetxt(euc_file_path, for_classification_euc, delimiter=',')