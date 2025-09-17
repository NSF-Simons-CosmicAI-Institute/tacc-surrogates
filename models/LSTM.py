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

# Import Pytorch LSTM module
from torch.nn import LSTM as lstm_base

# LSTM: Long Short Term Memory
class LSTM(torch.nn.Module):

	# Initialization
	def __init__(
		self,
		input_size: int,
		hidden_size: int,
		num_layers: int,
		n_epoch: int,
		batch_size: int
		):

		super().__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.n_epoch = n_epoch
		self.batch_size = batch_size

		self.model = lstm_base(
			input_size = self.input_size,
			hidden_size = self.hidden_size,
			num_layers = self.num_layers,
			batch_first = True)

	# Forward pass function
	# Note: x must be in the form (batch, seq length, features)
	# NOTE FROM LUKE: I'm pretty sure we need to map the features down as well, the output of LSTM is in the dimension of the hidden layer
	def forward(self, x):
		return self.model(x)	

		# Training function 
	def train(self, data_in, data_out):

		optimizer = optim.Adam(list(self.model.parameters()), lr = 1e-3)

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
				loss = torch.mean((y_pred-data_out[ind_batch])**2)
				loss.backward()
				optimizer.step()

			if it==0 or it%1==0:
				print('Epoch: ' + str(it) + '  |  ' + 'Loss: ' + str(float(loss.item())))

	# Evaluation function:
	def eval(self,x0):
		# x0 - the data point from which the prediction starts

		x_pred = self.forward(x0)
		return x_pred			

