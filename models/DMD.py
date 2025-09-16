# Import base python libraries
import math
import random
import os
import time

# Import the necessary array-handling libraries
import h5py
import torch
import numpy as np


# Dynamic Mode Decomposition 
class DMD(torch.nn.Module):

	# Initialization
	def __init__(
		self,
		n_modes: int,
		):

		super().__init__()

		self.n_modes = n_modes

	# Training function
	# NOTE: THERE ARE TWO INPUTS HERE TO THIS FUNCTION: YOUR "INPUT DATA" AND "OUTPUT DATA"
	# IN THAT SENSE, THE USER WILL FORMULATE THE X and Xp MATRICES
	# THIS CODE BASICALLY THEN JUST FINDS THE BEST FIT LINEAR MAP BETWEEN INPUTS AND OUTPUTS
	#
	# Very important:
	# data_in and data_out are assumed to be of shape: (time, space)
	def train(self, data_in, data_out):
		# data -- assumed to be a numpy array
		X = data_in.reshape(data_in.shape[0],-1)
		Xp = data_out.reshape(data_out.shape[0],-1)

		# Step 0: Pre-process the data matrix. Note that this step creates two data matrices: (1) the data matrix X at time t, and (2) the data matrix Xp at time t+dt.
		X = np.transpose(X,[1,0])
		Xp = np.transpose(Xp,[1,0])

		# Step 1: Compute the SVD of the dataset and truncate
		r = self.n_modes
		U,Sigma,VT = np.linalg.svd(X,full_matrices=False)
		Ur = U[:,:r]
		Sigmar = np.diag(Sigma[:r])
		VTr = VT[:r,:]

		# Step 2: Compute the transformed linear operator
		Atilde = np.linalg.solve(Sigmar.T,(Ur.T @ Xp @ VTr.T).T).T

		# Step 3: Eigen-decomposition of the transformed linear operator
		Lambda, W = np.linalg.eig(Atilde)
		Lambda = np.diag(Lambda)

		# Step 4: Compute DMD Modes
		Phi = Xp @ np.linalg.solve(Sigmar.T,VTr).T @ W
		alpha1 = Sigmar @ VTr[:,0]
		b = np.linalg.solve(W @ Lambda,alpha1)

		# Store the components of the DMD model
		self.Phi = Phi
		self.Lambda = Lambda

	# Evaluation Function
	def eval(self,x0,num_step):
		# x0 - the data point from which the prediction starts
		#    - assumed to be a single time step
		# num_step - the number of forward steps to predict
		xPred = np.zeros(((num_step,)+x0.shape))
		xPred[0] = x0.reshape(-1)

		# Reformat initial condition
		b = np.linalg.pinv(self.Phi)@xPred[0]

		# Loop through successive time steps
		for k in range(1,num_step):
			temp = self.Phi @ np.diag(np.exp(k*np.log(np.diag(self.Lambda)))) @ b
			xPred[k,:] = np.real(temp).reshape(x0.shape)

		# Return forward prediction
		return xPred	

    