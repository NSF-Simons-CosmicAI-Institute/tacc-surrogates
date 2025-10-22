#!/bin/bash

#SBATCH -J run-fno-training               # Job name
#SBATCH -o log.%j                         	 # Name of stdout output file (%j expands to jobId)
#SBATCH -p gpu-a100-dev                              # Queue name
#SBATCH -N 1                                 # Total number of nodes requested (56 cores/node)
#SBATCH -n 1                                 # Total number of mpi tasks requested
#SBATCH -t 02:00:00                          # Run time (hh:mm:ss)
#SBATCH -A XXXXXXXX							 # Project charge code

####SBATCH --mail-type=all
####SBATCH --mail-user=XXXXX@tacc.utexas.edu

# Load CUDA module
module load cuda/12.2

# Load tacc-surrogates libraries
source /scratch/10386/lsmith9003/python-envs/tacc-surrogates/bin/activate
export PYTHONPATH=$SCRATCH/scripts/tacc-surrogates

# Run flow bench training/evaluation
python tests/demo.py
