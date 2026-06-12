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

# Import PyTorch normalizing flows library
import normflows as nf
from normflows.flows.affine.coupling import Flow
from normflows.flows.reshape import Split, Merge

from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D


# Import base model methods
from models.Base import Base_Model as Base_Model

def kl_anneal(step, warmup_steps, cycle_steps):#ramp_steps):
    """
    Linear cyclical annealing.
    Beta ramps from 0 → 1 in each cycle.
    """
    max_beta = 1.0
    if step < warmup_steps:
    	beta = 0.0
    else:
    	x = step%cycle_steps
    	beta = pow(x/cycle_steps,1)#max_beta / (1.0 + math.exp(-sigmoid_k * (x - 0.5)))

    	#beta = min((step-100)/ramp_steps,1.0)

    return beta

# Custom function for splitting features and enforcing monotonicity on certain features
def split_mono(data,ind_mono,descending_bool):
    # Split feature indices
    ind_all = torch.arange(0,len(data[0,0,:]))
    mask = torch.ones_like(ind_all, dtype=torch.bool)
    mask[ind_mono] = False
    ind_reg = ind_all[mask]
    #data = torch.exp(data)
    data_mono  = data[:,:,ind_mono]
    data_reg = data[:,:,ind_reg]

    # Monotonicity transform on the relevant data
    # Assumes monotonically decreasing for now
    dim_mono = 1
    data_mono_temp = data_mono
    #data_mono = torch.flip(torch.cumsum(data_mono,dim=dim_mono),dims=[dim_mono])
    data_mono, _ = torch.sort(data_mono,dim=dim_mono,descending=descending_bool)

    # Re-concatenate 
    data_new = torch.zeros_like(data)
    data_new[:,:,ind_mono] = data_mono
    data_new[:,:,ind_reg] = data_reg

    # Log determinant contribution
    log_det = torch.sum(data_mono_temp,dim=list(range(1, data_mono_temp.dim())))

    return  data_new, log_det


# FNO: Fourier Neural Operator
class cVAE(Base_Model):

	# Initialization
	def __init__(
		self,
		n_modes: int,
		hidden_channels: int,
		num_prior: int,
		num_forward: int,
		num_features: int,
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
		self.num_features = num_features
		self.num_vector_components = num_vector_components

		self.activation = nn.ReLU()
		#self.activation = nn.Tanh()

		model = Unet1D(
			dim = 212,
			dim_mults = (1, 2, 4),
			channels = 536
			)

		self.diffusion = GaussianDiffusion1D(
			model,
			seq_length = 212,
			timesteps = 1000,
			objective = 'pred_v'
			)




		# Loss function
		self.mse = nn.MSELoss()



	# Forward pass function
	def forward(self, x):

		# Ignore initial condition for now, we will condition on it later.
		device = 'cuda' if torch.cuda.is_available() else 'cpu'

		# Encode conditioned on x0
		#x = torch.cat((x0,x),dim=1)
		x = torch.transpose(x,dim0=1,dim1=2)
		x = self.encoder(x)
		x = self.post_encoder(x)
		mu = x[:,0:self.latent_dim]
		logvar = x[:,self.latent_dim:]

		# Sample from P(z|x,x0)
		rand_temp = torch.randn(mu.shape)
		rand_temp = rand_temp.to(device)
		z = mu + rand_temp*torch.exp(0.50*logvar)

		# Condition decoder with x0
		z = torch.cat((z,torch.squeeze(x0)),dim=1)
		#z = self.pre_decoder(z)

		# Decode
		#z = torch.unsqueeze(z,dim=1)
		x = self.decoder(z)
		#x = torch.transpose(x,dim0=1,dim1=2)


		return x, mu, logvar


	# Data packing function
	def data_packing(self,data):
		if self.num_vector_components > 1:
			data_moved = torch.moveaxis(data,1,-1)
			data_flat = data_moved.reshape(*data_moved.shape[:-2],-1)
			data_packed = torch.moveaxis(data_flat,-1,1)
		else:
			data_packed = data
		return data_packed

	# Data un-packing function
	def data_unpacking(self,data):
		if self.num_vector_components > 1:
			data_moved = torch.moveaxis(data,1,-1)
			data_expanded = data_moved.reshape((*data_moved.shape[:-1], self.num_vector_components, self.num_forward))
			data_unpacked = torch.moveaxis(data_expanded,-1,1)
		else:
			data_unpacked = data
		return data_unpacked	

	# Training function 
	def train(self, data_in, data_out):

		print('Packing data...')
		data_in = self.data_packing(data_in)
		data_out = self.data_packing(data_out)
		sim_id = torch.arange(0,len(data_in))

		print('Setting up optimizer...')
		optimizer = optim.Adam([
			{'params': self.encoder.parameters()},
			{'params': self.decoder.parameters()},
			{'params': self.post_encoder.parameters()},
			{'params': self.pre_decoder.parameters()},
			#{'params': self.model_nf.parameters()},
			#{'params': self.preprocess_encoder.parameters()},
			],lr = self.learning_rate)

		print('Starting training...')
		for it in range(0, self.n_epoch):

			ind_shuffle = torch.randperm(data_in.size()[0])
			data_in = data_in[ind_shuffle]
			data_out = data_out[ind_shuffle]
			sim_id = sim_id[ind_shuffle]

			beta = kl_anneal(it,self.warmup_steps,self.cycle_steps)

			for ind in range(0,data_in.size()[0],self.batch_size):
				optimizer.zero_grad()
				ind_batch = range(ind,ind+self.batch_size)
				
				if ind+self.batch_size > data_in.size()[0]:
					ind_batch = range(ind,data_in.size()[0])

				y_pred, mu, logvar = self.forward(data_in[ind_batch],data_out[ind_batch])

				recon_loss = self.mse(y_pred,data_out[ind_batch])

				kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),dim=1)
				kl_loss = kl_loss.mean()

				loss = recon_loss + beta*kl_loss

				loss.backward()
				optimizer.step()

			print(
				'Epoch: ' + str(it) + '  |  ' + 'Loss: ' + str(float(loss.item()))
				)

	# Evaluation function:
	def eval(self,x0,x):
		# x0 - the data point from which the prediction starts
		x_in = self.data_packing(x0)
		x = self.data_packing(x)

		device = 'cuda' if torch.cuda.is_available() else 'cpu'
		x_pred = torch.zeros((len(x_in),self.num_forward,self.num_features))
		x_pred = x_pred.to(device)

		with torch.no_grad():
			for ind in range(0,len(x_in),self.batch_size):

				ind_batch = range(ind,ind+self.batch_size)

				if ind+self.batch_size > x_in.size()[0]:
					ind_batch = range(ind,x_in.size()[0])	

				x_temp = self.forward(x_in[ind_batch],x[ind_batch])
				x_pred[ind_batch] = x_temp
			
		x_out = self.data_unpacking(x_pred)
		return x_out			
			
	
