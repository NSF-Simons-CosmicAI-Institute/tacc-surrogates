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
from neuralop.models import FNO as FNO_Base

# Import base model methods
from models.Base import Base_Model as Base_Model

# FNO: Fourier Neural Operator
class FNO(Base_Model):

	# Initialization
	def __init__(
		self,
		n_modes: tuple,
		hidden_channels: int,
		num_prior: int,
		num_forward: int,
		num_vector_components: int,
		n_epoch: int,
		batch_size: int,
		learning_rate: float = 1e-3
		):

		super().__init__(n_epoch,batch_size,learning_rate)
		self.n_modes = n_modes
		self.hidden_channels = hidden_channels
		self.num_prior = num_prior
		self.num_forward = num_forward
		self.num_vector_components = num_vector_components

		self.model = FNO_Base(
			n_modes = self.n_modes,
			hidden_channels = self.hidden_channels,
			in_channels = self.num_prior*num_features,
			out_channels = self.num_forward*num_features)

		self.loss_function = nn.MSELoss()


	# Data packing function
	def data_packing(self,data):
		data_moved = torch.moveaxis(data,1,-1)
		data_flat = data_moved.reshape(*data_moved.shape[:-2],-1)
		data_packed = torch.moveaxis(data_flat,-1,1)
		return data_packed

	# Data un-packing function
	def data_unpacking(self,data):
		data_moved = torch.moveaxis(data,1,-1)
		data_expanded = data_moved.reshape((*data_moved.shape[:-1], self.num_features, self.num_forward))
		data_unpacked = torch.moveaxis(data_expanded,-1,1)
		return data_unpacked			
	
