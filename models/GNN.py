# Import base python libraries
import math
import random
import os
import time
import datetime
import functools
print = functools.partial(print, flush=True)

# Import the necessary array-handling libraries
import h5py
import numpy as np

# Import Pytorch for neural network training
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as dist

# Import GNN model(s) from Torch-Geometric
from torch_geometric.nn import GCNConv
from torch_geometric.utils import remove_self_loops


# Import base model methods
from models.Base import Base_Model as Base_Model

# FNO: Fourier Neural Operator
class GNN(Base_Model):

	# Initialization
	def __init__(
		self,
		hidden_channels: int,
		num_prior: int,
		num_forward: int,
		num_vector_components: int,
		n_epoch: int,
		batch_size: int,
		edge_matrix,
		edge_defintion: str = 'knn',
		activation: str = 'tanh',
		learning_rate: float = 1e-3
		):

		super().__init__(n_epoch,batch_size,learning_rate)
		self.num_vector_components = num_vector_components
		self.hidden_channels = hidden_channels
		self.num_prior = num_prior
		self.num_forward = num_forward
		if activation == 'relu':
			self.activation = nn.ReLU()
		elif activation == 'tanh':
			self.activation = nn.Tanh()
		self.edge_matrix = edge_matrix	

		self.conv1 = GCNConv(self.num_prior*self.num_vector_components,self.hidden_channels)
		self.conv2 = GCNConv(self.hidden_channels,self.num_forward*self.num_vector_components)

		self.mean = nn.Sequential(nn.Conv2d(1,10,kernel_size=1),self.activation,nn.Conv2d(10,1,kernel_size=1))
		self.log_std = nn.Sequential(nn.Conv2d(1,10,kernel_size=1),self.activation,nn.Conv2d(10,1,kernel_size=1))

		self.node_mix1 = nn.Sequential(nn.Linear(160,1024),self.activation,nn.Linear(1024,160))
		self.node_mix2 = nn.Sequential(nn.Linear(160,1024),self.activation,nn.Linear(1024,160))

		self.loss_function = nn.MSELoss()

	# Forward pass function
	def forward(self, x, edge_matrix, sim_id):

		x = self.node_mix1(x)

		x = torch.unsqueeze(x,1)
		mean = self.mean(x)
		log_std = self.log_std(x)
		mean = torch.squeeze(mean,1)
		log_std = torch.squeeze(log_std,1)

		dist_normal = dist.Normal(mean,torch.exp(log_std))
		x = []
		for batch_ind in range(0,len(sim_id)):
			torch.manual_seed(sim_id[batch_ind])
			temp = dist_normal.rsample()
			x.append(temp[0].tolist())	
		x = torch.tensor(x,dtype=torch.float32)

		x = self.node_mix2(x)

		x = self.conv1(x,edge_matrix)
		x = self.activation(x)
		x = self.conv2(x,edge_matrix)
		return x	

	# Data packing function
	def data_packing(self,data):
		if self.num_vector_components > 1:
			data_moved = torch.moveaxis(data,1,-1)
			data_flat = data_moved.reshape(*data_moved.shape[:-2],-1)
			data_packed = data_flat.reshape((data_flat.shape[0],np.prod(data_flat.shape[1:-1]),data_flat.shape[-1]))
			self.original_grid_dimension = data_flat.shape[1:-1]

		else:
			data_moved = torch.moveaxis(data,1,-1)
			data_packed = data_moved.reshape((data_moved.shape[0],np.prod(data_moved.shape[1:-1]),data_moved.shape[-1]))
			self.original_grid_dimension = data_moved.shape[1:-1]
		return data_packed

	# Data un-packing function
	def data_unpacking(self,data):
		if self.num_vector_components > 1:
			data_expanded = data.reshape((*data.shape[:-1], self.num_vector_components, self.num_forward))
			data_moved = torch.moveaxis(data_expanded,-1,1)
			data_unpacked = data_moved.reshape((*data_moved.shape[0:2],*self.original_grid_dimension,data_moved.shape[-1]))
		else:
			data_moved = torch.moveaxis(data_expanded,-1,1)
			data_unpacked = data_moved.reshape((*data_moved.shape[0:2],*self.original_grid_dimension))
		return data_unpacked	

	# Training function 
	def train(self, data_in, data_out):

		print('Packing data...')
		data_in = self.data_packing(data_in)
		data_out = self.data_packing(data_out)
		sim_id = torch.arange(0,len(data_in))

		print('Setting up optimizer...')
		#optimizer = optim.Adam(list(self.conv1.parameters())+list(self.conv2.parameters())+[self.edge_matrix], lr = self.learning_rate)
		optimizer = optim.Adam([
			{'params': self.conv1.parameters()},
			{'params': self.conv2.parameters()},	
			{'params': self.mean.parameters()},
			{'params': self.log_std.parameters()},
			{'params': self.node_mix1.parameters()},
			{'params': self.node_mix2.parameters()},
			],lr = self.learning_rate)

		print('Setting up edge matrix...')
		edges = []
		for ind0 in range(0,len(self.edge_matrix[:,0])):
			for ind1 in range(0,len(self.edge_matrix[0,:])):
				if self.edge_matrix[ind0,ind1]!=0:
					edges.append([ind0,ind1])
		edges = torch.tensor(edges,dtype=torch.int32)
		edges = torch.transpose(edges,1,0)	
		self.edge_matrix = edges		
	

		#node_num = data_in.shape[1]
		#edge_matrix = torch.zeros(2,node_num*2)#node_num)
		#for ind in range(0,node_num):
		#	temp = ind*torch.ones(2,2)#node_num)
		#	temp[1,:] = torch.tensor([ind-1,ind+1])
		#	if ind ==0:
		#		temp[1,:] = torch.tensor([ind+1,ind+2])
		#	elif ind==(node_num-1):
		#		temp[1,:] = torch.tensor([ind-1,ind-2])

		#	edge_matrix[:,ind*2:(ind+1)*2] =  temp
			
		#self.edge_matrix = edge_matrix.to(torch.int32)
		#self.edge_matrix = remove_self_loops(self.edge_matrix)[0]

		#print(self.edge_matrix)

		print('Starting training...')
		for it in range(0, self.n_epoch):

			ind_shuffle = torch.randperm(data_in.size()[0])
			data_in = data_in[ind_shuffle]
			data_out = data_out[ind_shuffle]
			sim_id = sim_id[ind_shuffle]

			for ind in range(0,data_in.size()[0],self.batch_size):
				optimizer.zero_grad()
				ind_batch = range(ind,ind+self.batch_size)
				
				if ind+self.batch_size > data_in.size()[0]:
					ind_batch = range(ind,data_in.size()[0])

				y_pred = self.forward(data_in[ind_batch],self.edge_matrix,sim_id[ind_batch])
				loss = self.loss_function(y_pred,data_out[ind_batch])

				#mse = (y_pred-data_out[ind_batch])**2
				#zero_mask = (data_out[ind_batch] == 0).float()
				#nonzero_mask = (data_out[ind_batch] != 0).float()
				#loss = (0.01 * zero_mask + 1.0 * nonzero_mask) * mse
				#loss = torch.mean(loss)

				loss.backward()
				optimizer.step()
			print('Epoch: ' + str(it) + '  |  ' + 'Loss: ' + str(float(loss.item())))

	# Evaluation function:
	def eval(self,x0):
		# x0 - the data point from which the prediction starts
		sim_id = torch.arange(0,len(x0))
		x_in = self.data_packing(x0)
		x_pred = self.forward(x_in,self.edge_matrix,sim_id)
		x_out = self.data_unpacking(x_pred)
		return x_out	
