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
from torch.nn import LSTM as LSTM_Base

# Import base model methods
from models.Base import Base_Model as Base_Model


# LSTM: Long Short Term Memory
class LSTM(Base_Model):

	# Initialization
	def __init__(
		self,
		input_size: int,
		hidden_size: int,
		num_layers: int,
		n_epoch: int,
		batch_size: int,
		learning_rate: float = 1e-3
		):

		super().__init__(n_epoch,batch_size,learning_rate)
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers

		self.model = LSTM_Base(
			input_size = self.input_size,
			hidden_size = self.hidden_size,
			num_layers = self.num_layers,
			batch_first = True)

		self.loss_function = nn.MSELoss()

	# Forward pass function
	# Note: x must be in the form (batch, seq length, features)
	# NOTE FROM LUKE: I'm pretty sure we need to map the features down as well, the output of LSTM is in the dimension of the hidden layer
	def forward(self, x):
		return self.model(x)	



