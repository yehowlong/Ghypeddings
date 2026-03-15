import torch as th
import torch.nn as nn
import torch.nn.functional as F
from HGNN.utils import * 
from HGNN.hyperbolic_module.CentroidDistance import CentroidDistance

class NodeClassification(nn.Module):

	def __init__(self, args, logger, rgnn, manifold):
		super(NodeClassification, self).__init__()
		self.args = args
		self.logger = logger
		self.manifold = manifold
		self.c = nn.Parameter(th.Tensor([1.]))

		self.feature_linear = nn.Linear(self.args.input_dim,
										self.args.dim
							  )
		nn_init(self.feature_linear, self.args.proj_init)
		self.args.eucl_vars.append(self.feature_linear)			

		self.distance = CentroidDistance(args, logger, manifold)

		self.rgnn = rgnn
		self.output_linear = nn.Linear(self.args.num_centroid,
										self.args.num_class
							  )
		nn_init(self.output_linear, self.args.proj_init)
		self.args.eucl_vars.append(self.output_linear)

		self.log_softmax = nn.LogSoftmax(dim=1)
		self.activation = get_activation(self.args)

	def forward(self, adj, weight, features):

		adj, weight, features = adj.squeeze(0), weight.squeeze(0), features.squeeze(0)
		node_repr = self.activation(self.feature_linear(features))
		assert th.isnan(node_repr).any().item() == False
		mask = th.ones((self.args.node_num, 1)).cuda() # [node_num, 1]
		node_repr = self.rgnn(node_repr, adj, weight, mask) # [node_num, embed_size]

		_, node_centroid_sim = self.distance(node_repr, mask) # [1, node_num, num_centroid]
		class_logit = self.output_linear(node_centroid_sim.squeeze())
		return self.log_softmax(class_logit) , node_repr