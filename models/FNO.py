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
from Base import Base_Model as Base_Model

# FNO: Fourier Neural Operator
class FNO(Base_Model):

	# Initialization
	def __init__(
		self,
		n_modes: tuple,
		hidden_channels: int,
		in_channels: int,
		out_channels: int,
		):

		super().__init__()
		self.n_modes = n_modes
		self.hidden_channels = hidden_channels
		self.in_channels = in_channels
		self.out_channels = out_channels

		self.model = FNO_Base(
			n_modes = self.n_modes,
			hidden_channels = self.hidden_channels,
			in_channels = self.in_channels,
			out_channels = self.out_channels)

		self.loss_function = nn.MSELoss()
