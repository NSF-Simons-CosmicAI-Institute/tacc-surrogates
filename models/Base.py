# Import base python libraries
import math
import random
import os
import time

# Import the necessary array-handling libraries
import h5py
import numpy as np

# Import Pytorch for neural network training
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Base Model: template class that includes methods common to all architectures
class Base_Model(torch.nn.Module):

	# Initialization
	def __init__(
		self,
		n_epoch: int,
		batch_size: int,
		learning_rate: float = 1e-3
		):

		super().__init__()
		self.n_epoch = n_epoch
		self.batch_size = batch_size
		self.learning_rate = learning_rate

		self.model = nn.Identity()
		self.loss_function = nn.MSELoss()


	# Forward pass function
	def forward(self, x):
		return self.model(x)	

	# Training function 
	def train(self, data_in, data_out):

		optimizer = optim.Adam(list(self.model.parameters()), lr = self.learning_rate)

		for it in range(0, self.n_epoch):

			ind_shuffle = torch.randperm(data_in.size()[0])
			data_in = data_in[ind_shuffle]
			data_out = data_out[ind_shuffle]

			for ind in range(0,data_in.size()[0],self.batch_size):
				optimizer.zero_grad()
				ind_batch = range(ind,ind+self.batch_size)
				
				if ind+self.batch_size > data_in.size()[0]:
					ind_batch = range(ind,data_in.size()[0])

				y_pred = self.forward(data_in[ind_batch])
				loss = self.loss_function(y_pred,data_out[ind_batch])
				loss.backward()
				optimizer.step()

			if it==0 or it%1==0:
				print('Epoch: ' + str(it) + '  |  ' + 'Loss: ' + str(float(loss.item())))

	# Evaluation function:
	def eval(self,x0):
		# x0 - the data point from which the prediction starts
		x_pred = self.forward(x0)
		return x_pred			