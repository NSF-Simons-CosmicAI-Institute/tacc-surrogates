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

# Import PyTorch neural operator library
from neuralop.models import FNO as fno_base

# FNO: Fourier Neural Operator
class FNO(torch.nn.Module):

	# Initialization
	def __init__(
		self,
		n_modes: tuple,
		hidden_channels: int,
		in_channels: int,
		out_channels: int,
		n_epoch: int,
		batch_size: int
		):

		super().__init__()
		self.n_modes = n_modes
		self.hidden_channels = hidden_channels
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.n_epoch = n_epoch
		self.batch_size = batch_size

		self.model = fno_base(
			n_modes = self.n_modes,
			hidden_channels = self.hidden_channels,
			in_channels = self.in_channels,
			out_channels = self.out_channels)

	# Forward pass function
	def forward(self, x):
		return self.model(x)	

	# Training function 
	def train(self, data_in, data_out):

		print('Setting up optimizer...')
		optimizer = optim.Adam(list(self.model.parameters()), lr = 1e-3)
		print('Done.')

		for it in range(0, self.n_epoch):

			print('Shuffling data...')
			ind_shuffle = torch.randperm(data_in.size()[0])
			data_in = data_in[ind_shuffle]
			data_out = data_out[ind_shuffle]
			print('Done.')

			for ind in range(0,data_in.size()[0],self.batch_size):
				optimizer.zero_grad()
				ind_batch = range(ind,ind+self.batch_size)
				now = datetime.datetime.now()
				print('Batch number: ' + str(ind))
				print(now)
				
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
		#    - assumed to be a single time step

		x_pred = self.forward(x0)
		return x_pred	
