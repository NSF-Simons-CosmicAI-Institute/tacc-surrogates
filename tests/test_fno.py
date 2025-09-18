# Libraries
import functools
import os
import numpy as np
import torch
from scipy.io import loadmat
from FNO import FNO
from utils import get_re_load, clean_subdir_array
from matplotlib import pyplot as plt

# Custom print
print = functools.partial(print, flush=True)

# Reynolds number (target)
re_target = 200.

# FNO Specifications (num prior time steps + num forward time steps)
num_prior = 4
num_forward = 10

# Root directory
root = '/scratch/10386/lsmith9003/data/FlowBench/FPO_NS_2D_1024x256/harmonics/'
subdir_array = np.arange(1,10,1)
subdir_array = clean_subdir_array(root,subdir_array)
print('allocating data arrays')

# Intialize datasets
data_in = np.zeros((len(subdir_array)*(242-num_prior-num_forward),2+3*num_prior,256,1024))
data_out = np.zeros((len(subdir_array)*(242-num_prior-num_forward),3*num_forward,256,1024))
print('allocation done')
print('loading data')

# Load dataset
count = 0
for subdir in subdir_array:
	dir_path = root + str(subdir)
	re_load = get_re_load(re_target,dir_path)

	print('Loading data for path: ' + dir_path + ' Re = ' + str(re_load))

	data_flow = np.load(root + str(subdir) + '/Re_' + str(re_load) + '.npz')['data']
	data_geom = np.load(root + str(subdir) + '/input_geometry.npz')
	for ind in range(num_prior,242-num_forward):
		data_in[count,0,:,:] = data_geom['data']
		data_in[count,1,:,:] = data_geom['mask']
		data_in[count,2:(2+num_prior),:,:] = np.squeeze(data_flow[ind-num_prior:ind,:,:,0])
		data_in[count,(2+num_prior):(2+2*num_prior),:,:] = np.squeeze(data_flow[ind-num_prior:ind,:,:,1])
		data_in[count,(2+2*num_prior):(2+3*num_prior),:,:] = np.squeeze(data_flow[ind-num_prior:ind,:,:,2])

		data_out[count,0:num_forward,:,:] = np.squeeze(data_flow[ind:ind+num_forward,:,:,0])
		data_out[count,num_forward:2*num_forward,:,:] = np.squeeze(data_flow[ind:ind+num_forward,:,:,1])
		data_out[count,2*num_forward:3*num_forward,:,:] = np.squeeze(data_flow[ind:ind+num_forward,:,:,2])
		count += 1

# Convert to tensor
data_in = torch.tensor(data_in,dtype=torch.float32)
data_out = torch.tensor(data_out,dtype=torch.float32)

# Initialize FNO
print('initializing FNO')
fno_test = FNO(
	n_modes=(16,16),
	hidden_channels=32,
	in_channels=(2+3*num_prior),
	out_channels=(3*num_forward),
	n_epoch=10,
	batch_size=16
	)
print('done')

# Train
#print('starting training')
#fno_test.train(data_in,data_out)
#print('done')

# Save
print('Saving model...')
torch.save(fno_test,'fno_flowbench.pth')
print('done')

# Eval
print('Evaluating...')
fno_test = torch.load('fno_flowbench.pth',weights_only=False)
x0 = data_in[0:2]
data_pred = fno_test.eval(x0)

# Plot time series prediction
data_pred = data_pred[0]
u_pred = data_pred[0:10]
v_pred = data_pred[10:20]
p_pred = data_pred[20:30]
u_true = data_out[0][0:10]
v_true = data_out[0][10:20]
p_true = data_out[0][20:30]
for time_ind in range(0,len(u_pred)):
    plt.figure(figsize=(10,2))
    plt.subplot(1,2,1)
    plt.pcolormesh(np.squeeze(u_pred[time_ind].detach().numpy()),vmin=-1,vmax=2,cmap='RdBu')
    plt.xticks([])
    plt.yticks([])
    plt.title('Predicted')
    plt.subplot(1,2,2)
    plt.pcolormesh(np.squeeze(u_true[time_ind].detach().numpy()),vmin=-1,vmax=2,cmap='RdBu')
    plt.xticks([])
    plt.yticks([])
    plt.title('True')
    plt.savefig("figures/fig"+str(time_ind)+".png")
print('figures saved')
print(np.shape(data_pred))
print('done')
