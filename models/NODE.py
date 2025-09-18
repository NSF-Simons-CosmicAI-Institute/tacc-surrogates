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

# Import PyTorch differential equation library
from torchdiffeq import odeint

# Import base model methods
from Base import Base_Model as Base_Model

# NODE: Neural Ordinary Differential Equation
class NODE(Base_Model):

	# Initialization:
	def __init__(
		self,
		n_channel: int,
		kernel_size: int = 5,
		padding: str = 'same'
		):

			super().__init__()

			self.n_channel = n_channel
			self.kernel_size = kernel_size
			self.padding = padding

			# Neural network definition
			self.ode = nn.Sequential(
				nn.Conv2d(self.n_channel,2*self.n_channel,kernel_size=self.kernel_size,stride=1,padding=self.padding),
				nn.Tanh(),
				nn.Conv2d(2*self.n_channel,4*self.n_channel,kernel_size=self.kernel_size,stride=1,padding=self.padding),
				nn.Tanh(),
				nn.Conv2d(4*self.n_channel,2*self.n_channel,kernel_size=self.kernel_size,stride=1,padding=self.padding),
				nn.Tanh(),
				nn.Conv2d(2*self.n_channel,self.n_channel,kernel_size=self.kernel_size,stride=1,padding=self.padding)
				)

	# Forward pass function:
	def forward(self, t, x):
		return self.ode(x)		

	# Training function:
	# IT WOULD BE VERY NICE IF WE COULD ALSO A VECTOR THAT SPECIFIES TIME STEP
	# SEPARATION BETWEEN INPUTS AND OUTPUTS. THIS WAY WE COULD HAVE MULTIPLE 
	# TIME STEP SEPARATIONS IN THE INPUT/OUTPUT DATA.
	#
	# CURRENTLY SET UP TO HANDLE DATA AT ADJACENT TIME STEPS
	def train(self, data_in, data_out):

		optimizer = optim.Adam(list(self.ode.parameters()), lr = 1e-3)

		for it in range(0, self.n_epoch):

			ind_shuffle = torch.randperm(data_in.size()[0])
			data_in = data_in[ind_shuffle]
			data_out = data_out[ind_shuffle]

			for ind in range(0,data_in.size()[0],batch_size):
				optimizer.zero_grad()
				ind_batch = range(ind,ind+batch_size)

				if ind+batch_size > data_in.size()[0]:
					ind_batch = range(ind,data_in.size()[0])

				y_pred = odeint(self,data_in[ind_batch],torch.arange(0,2,dtype=torch.float32))
				y_pred = torch.squeeze(y_pred[1:])
				loss = torch.mean((y_pred-data_out[ind_batch])**2)
				loss.backward()
				optimizer.step()
			
			if it==0 or it%10==0:
				print('Epoch: ' + str(it) + '  |  ' + 'Loss: ' + str(float(loss.item())))

	# Evaluation function:
	def eval(self,x0,num_steps):
		# x0 - the data point from which the prediction starts
		#    - assumed to be a single time step
		# num_step - the number of forward steps to predict

		x_pred = odeint(self,x0,torch.arange(0,num_steps,dtype=torch.float32))
		return x_pred



