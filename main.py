# Libraries
import functools
import os
import numpy as np
import pandas as pd
import torch
import glob
import time
import json
from matplotlib import pyplot as plt
from models.EPI import EPI 


# Input parameters
state_name = 'New-Jersey' # options: District-of-Columbia, New-Jersey, North-Carolina, North-Dakota, Wisconsin
R0 = [1.0, 3.0, 5.0]
# We'll need a new way of parsing the initial infected json here, it's particular

# Simulation constants
num_features = 4
num_cases_per_dataset = 100
num_timesteps_total = 500
num_timesteps_prior = 1
num_timesteps_forward = num_timesteps_total - num_timesteps_prior

# Relevant file paths
metadata_master_path = '/scratch/10386/lsmith9003/data/Epi_Surrogate_Modeling/metadata_master.csv'
county_pop_path = '/scratch/10386/lsmith9003/data/Epi_Surrogate_Modeling/data/' + state_name + '/county_pop_by_age_' + state_name + '_2019-2023ACS.csv'
edge_matrix_path = '/scratch/10386/lsmith9003/data/Epi_Surrogate_Modeling/data/' + state_name + '/' + state_name + '_Q4-2019_mobility-matrix.csv'
data_dir = '/scratch/10386/lsmith9003/data/Epi_Surrogate_Modeling/SEIR-STOCH_Param_Sweep/' + state_name +'/'

# CUDA check
print('Checking Cuda...')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(torch.cuda.is_available())
print(f"Using device: {device}")


# -------------------- Data Loading -------------------- #

# Status update
print('Loading data...')

# Read metadata mater file
data = pd.read_csv(metadata_master_path)

# Simple for loop over rows
# Note that we need the 'sim_completion' flag to be checked here. Emily reran cases for which 100 simulations did not finish.
count = 0
num_nodes = 0
output_dirs = []

for ind in range(0,data.shape[0]):
	if data.loc[ind,'disease_R0'] in R0 and \
	data.loc[ind,'geo_region'] == state_name and \
	data.loc[ind,'sim_completion'] == 1:

		# Read initial infected json as a list of dictionaries
		# It should be read like: data_infected[i]['age_group']
		# where i is the seeded county, and the second key can be 'county', 'infected', or 'age_group'
		# Note that some cases seed multiple counties, so i will not always be 0.
		# For our initial subset, however, we're only looking for cases with a single seeded county, so len(data_infected)==1
		data_infected = json.loads(data.loc[ind,'initial_infected_json'])
		if len(data_infected) == 1.:
			output_dirs.append(data.loc[ind,'output_dir_path'])
			count += 1
			if not num_nodes:
				num_nodes = data.loc[ind,'geo_node_count']

# Initialize 
data_in = np.zeros((len(output_dirs)*num_cases_per_dataset,num_timesteps_prior,num_nodes,num_features))
data_out = np.zeros((len(output_dirs)*num_cases_per_dataset,num_timesteps_forward,num_nodes,num_features))

# Loop through identified files and load contents
case_count = 0
for output_dir in output_dirs:
	if case_count % 10 == 0 or case_count == len(output_dirs):
		print('Loading data for case: ' + str(case_count) + ' of ' + str(len(output_dirs)))
	case_dir = data_dir + output_dir
	files = glob.glob(os.path.join(case_dir,'node_*_batch-*.csv'))
	node_count = 0
	for file in files:
		data_sim = pd.read_csv(file).to_numpy()
		sim_id = data_sim[:,0]
		time_id = data_sim[:,1]
		data_sim = data_sim[:,5:(5+num_features)]
		for ind in range(0,num_cases_per_dataset):
			sim_inds = np.where(sim_id==ind)[0]
			data_ind = data_sim[sim_inds]
			if len(sim_inds) < num_timesteps_total:
				temp_tile = np.tile(data_ind[-1],(num_timesteps_total-len(sim_inds),1))
				data_ind = np.concatenate((data_ind,temp_tile))
			data_in[int(case_count*num_cases_per_dataset+ind),:,int(node_count),:] = data_ind[0:num_timesteps_prior]
			data_out[int(case_count*num_cases_per_dataset+ind),:,int(node_count),:] = data_ind[num_timesteps_prior:(num_timesteps_prior+num_timesteps_forward)]	
		node_count += 1
	case_count += 1

# Convert to tensor
data_in = torch.tensor(data_in,dtype=torch.float32)
data_out = torch.tensor(data_out,dtype=torch.float32)

# Move to GPU
data_in = data_in.to(device)
data_out = data_out.to(device)



# -------------------- Normalization -------------------- #

# Load county population matrix + sum over age group
county_pop = pd.read_csv(county_pop_path).to_numpy()
county_pop = np.sum(county_pop[:,1:],axis=-1)

# # Load edge matrix
edge_matrix = pd.read_csv(edge_matrix_path,skip_blank_lines=False,header=None).to_numpy()
# # We'll skip normalizing the edge weights for now, the max is close enough to 1 that we can revisit later.

# Convert edge matrix to format for GNN
edge_index = np.zeros((2,num_nodes*num_nodes))
edge_weight = np.zeros((num_nodes*num_nodes,))
count = 0
for ind_i in range(0,num_nodes):
	for ind_j in range(0,num_nodes):
 		edge_index[:,count] = [ind_i,ind_j]
 		edge_weight[count] = edge_matrix[ind_i,ind_j]
 		count = count + 1
# Convert to tensor
edge_index = torch.tensor(edge_index,dtype=torch.int32)
edge_weight = torch.tensor(edge_weight,dtype=torch.float32)

# # Move to GPU
edge_index = edge_index.to(device)
edge_weight = edge_weight.to(device)



# -------------------- Model Initialization -------------------- #

# Initialize model architecture
print('Initializing model...')
epi = EPI(
	edge_index=edge_index,
	edge_weight=edge_weight,
	county_pop=county_pop,
	hidden_channels=64,
	n_epoch= 5,
	batch_size=16,
	learning_rate = 1e-3,
	num_prior=num_timesteps_prior,
	num_total_timesteps=num_timesteps_forward,
	num_forward=1,
	num_features=num_features,
	num_nodes = num_nodes
	)

epi = epi.to(device)



# -------------------- Model Training -------------------- #


# Status update
# print('Training...')

# Load existing pre-trained model (Optional)
# epi = torch.load('fno_epidemic.pth',weights_only=False)
# epi.n_epoch = 30

# Train model
epi.train(data_in,data_out)

# Save final model
print('Saving final model...')
torch.save(epi,'epi.pth')
print('done')




# -------------------- Model Evaluation -------------------- #


# Status update
print('Evaluating...')

# Load model weights
epi = torch.load('epi.pth',weights_only=False)

# Define input samples for evaluation (random)
ind_sample = np.random.randint(low=0, high=len(data_in), size=100)
sample = data_in[ind_sample]

# Evaluate samples
start_time = time.perf_counter()
data_pred = epi(sample)
end_time = time.perf_counter()
print('Size of evaluated dataset:')
print(data_pred.shape)
print('Evaluation time:')
print(str(end_time-start_time)+' seconds')
print('Done.')

# Collapse results into a global metric
data_pred = data_pred.cpu().detach().numpy()
data_out = data_out.cpu().detach().numpy()
data_pred = np.sum(data_pred[:,:,:,0],axis=2)
data_out = np.sum(data_out[:,:,:,0],axis=2)

# Plot time series prediction (all simulations)
plt.figure(figsize=(15,5))
ybounds = [1e6,1e7]
for sim_ind in range(0,len(data_pred)):
    plt.subplot(1,3,1)
    plt.plot(data_pred[sim_ind,:],alpha=0.5)
    plt.xlabel('Simulation Day')
    plt.ylabel('Susceptible')
    plt.title('Prediction')
    plt.ylim(ybounds)
    plt.subplot(1,3,2)
    plt.plot(data_out[np.random.randint(low=0,high=len(data_out)),:],alpha=0.5)
    plt.xlabel('Simulation Day')
    plt.ylabel('Susceptible')
    plt.title('Truth')
    plt.ylim(ybounds)
    if sim_ind == 0:
        plt.subplot(1,3,3)
        plt.plot(np.mean(data_out[ind_sample,:],axis=0),color=(0.3,0.3,0.3))
        plt.plot(np.mean(data_out[ind_sample,:],axis=0)+np.std(data_out[ind_sample,:],axis=0),linestyle=':',color=(0.3,0.3,0.3))
        plt.plot(np.mean(data_out[ind_sample,:],axis=0)-np.std(data_out[ind_sample,:],axis=0),linestyle=':',color=(0.3,0.3,0.3))

        plt.plot(np.mean(data_pred,axis=0),color=(0.6,0.74,0.92))
        plt.plot(np.mean(data_pred,axis=0)+np.std(data_pred,axis=0),linestyle=':',color=(0.6,0.74,0.92))
        plt.plot(np.mean(data_pred,axis=0)-np.std(data_pred,axis=0),linestyle=':',color=(0.6,0.74,0.92))
        plt.xlabel('Simulation Day')
        plt.ylabel('Susceptible')
        plt.title('Distribution Stats')
        plt.ylim(ybounds)
plt.savefig("figures/susceptible.png")
print('figures saved')
print(np.shape(data_pred))
print('done')


