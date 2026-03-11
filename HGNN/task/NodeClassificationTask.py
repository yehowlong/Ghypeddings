import torch as th
import torch.nn as nn
import torch.nn.functional as F
from Ghypeddings.HGNN.utils import * 
from torch.utils.data import DataLoader
import torch.optim as optim
from Ghypeddings.HGNN.task.BaseTask import BaseTask
import numpy as np
from Ghypeddings.HGNN.dataset.NodeClassificationDataset import NodeClassificationDataset
from Ghypeddings.HGNN.task.NodeClassification import NodeClassification
import time
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,precision_score,recall_score

def cross_entropy(log_prob, label, mask):
	label, mask = label.squeeze(), mask.squeeze()
	negative_log_prob = -th.sum(label * log_prob, dim=1)
	return th.sum(mask * negative_log_prob, dim=0) / th.sum(mask)

def get_accuracy(label, log_prob, mask):
	lab = label.clone()
	lab = lab.squeeze()
	mask_copy = mask.clone().cpu().numpy()[0].astype(np.bool_)
	pred_class = th.argmax(log_prob, dim=1).cpu().numpy()[mask_copy]
	real_class = th.argmax(lab, dim=1).cpu().numpy()[mask_copy]
	acc= accuracy_score(y_true=real_class,y_pred=pred_class)
	f1= f1_score(y_true=real_class,y_pred=pred_class)
	recall= recall_score(y_true=real_class,y_pred=pred_class)
	precision= precision_score(y_true=real_class,y_pred=pred_class)
	print(np.sum(real_class) , np.sum(pred_class))
	roc_auc = roc_auc_score(real_class,pred_class)	
	return acc,f1,recall,precision,roc_auc

class NodeClassificationTask(BaseTask):

	def __init__(self, args, logger, rgnn, manifold,adj,features,labels):
		super(NodeClassificationTask, self).__init__(args, logger, criterion='max')
		self.args = args
		self.logger = logger
		self.manifold = manifold
		self.hyperbolic = True
		self.rgnn = rgnn
		self.loader = self.process_data(adj,features,labels)
		self.model = NodeClassification(self.args, self.logger, self.rgnn, self.manifold).cuda()
		self.loss_function = cross_entropy

	def forward(self, model, sample, loss_function):
		scores , embeddings = model(
					sample['adj'].cuda().long(),
			        sample['weight'].cuda().float(),
			        sample['features'].cuda().float(),
					)
		loss = loss_function(scores,
						 sample['y_train'].cuda().float(), 
						 sample['train_mask'].cuda().float())
		return scores, loss , embeddings

	def run_gnn(self):
		loader = self.loader
		model = self.model
		loss_function = self.loss_function
		
		self.args.manifold = self.manifold
		optimizer, lr_scheduler, hyperbolic_optimizer, hyperbolic_lr_scheduler = \
								set_up_optimizer_scheduler(self.hyperbolic, self.args, model,self.manifold)
		self.labels = None
		
		best_losses = []
		train_losses = []
		val_losses = []

		t_total = time.time()
		for epoch in range(self.args.epochs):
			model.train()
			for i, sample in enumerate(loader):
				model.zero_grad()
				scores, loss , embeddings = self.forward(model, sample, loss_function)
				loss.backward()
				if self.args.grad_clip > 0.0:
					th.nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_clip)
				optimizer.step()
				if self.hyperbolic and len(self.args.hyp_vars) != 0:
					hyperbolic_optimizer.step()
				self.labels = sample['y_train']
				accuracy,f1,recall,precision,roc_auc = get_accuracy(
									sample['y_train'].cuda().float(), 
									scores, 
									sample['train_mask'].cuda().float())
			
				train_losses.append(loss.item())
				if(len(best_losses) == 0):
					best_losses.append(train_losses[0])
				elif (best_losses[-1] > train_losses[-1]):
					best_losses.append(train_losses[-1])
				else:
					best_losses.append(best_losses[-1])

				if (epoch + 1) % self.args.log_freq == 0:
					self.logger.info("%s epoch %d: accuracy %.4f f1 %.4f recall %.4f precision %.4f roc_auc %.4f loss: %.4f \n" % (
						'train', 
						epoch, 
						accuracy,f1,recall,precision,roc_auc,loss.item()))
					
				dev_loss, accuracy ,f1,recall,precision,roc_auc  = self.evaluate(loader, 'val', model, loss_function)
				val_losses.append(dev_loss)
				lr_scheduler.step()

				if self.hyperbolic and len(self.args.hyp_vars) != 0:
					hyperbolic_lr_scheduler.step()
				if not self.early_stop.step(dev_loss, epoch , embeddings):		
					break

		self.logger.info("Training Finished!")
		self.logger.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))

		return {'train':train_losses,'best':best_losses,'val':val_losses}, accuracy,f1,recall,precision,roc_auc,time.time() - t_total
			
	def evaluate(self, data_loader, prefix, model, loss_function):
		model.eval()
		with th.no_grad():
			for i, sample in enumerate(data_loader):
				scores, loss , _ = self.forward(model, sample, loss_function)
				if prefix == 'val':
					accuracy,f1,recall,precision,roc_auc = get_accuracy(
									sample['y_val'].cuda().float(), 
									scores, 
									sample['val_mask'].cuda().float())
				elif prefix == 'test':
					accuracy,f1,recall,precision,roc_auc = get_accuracy(
									sample['y_test'].cuda().float(), 
									scores, 
									sample['test_mask'].cuda().float())
				
		return loss.item(), accuracy,f1,recall,precision,roc_auc

	def process_data(self,adj,features,labels):
		dataset = NodeClassificationDataset(self.args, self.logger,adj,features,labels)
		return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
