import numpy as np
from Ghypeddings.HGNN.utils import *
import torch as th
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

class BaseTask(object):
	"""
	A base class that supports loading datasets, early stop and reporting statistics
	"""
	def __init__(self, args, logger, criterion='max'):
		"""
		criterion: min/max
		"""
		self.args = args
		self.logger = logger
		self.early_stop = EarlyStoppingCriterion(self.args.patience, criterion)

	def reset_epoch_stats(self, epoch, prefix):
		"""
		prefix: train/dev/test
		"""
		self.epoch_stats = {
			'prefix': prefix,
			'epoch': epoch,
			'loss': 0,
			'num_correct': 0,
			'num_total': 0,
		}

	def update_epoch_stats(self, loss, score, label, is_regression=False):
		with th.no_grad():
			self.epoch_stats['loss'] += loss.item()
			self.epoch_stats['num_total'] += label.size(0)
			if not is_regression:
				self.epoch_stats['num_correct'] += th.sum(th.eq(th.argmax(score, dim=1), label)).item()
	
	def report_best(self):
		self.logger.info("best val %.6f" 
			% (self.early_stop.best_dev_score))
