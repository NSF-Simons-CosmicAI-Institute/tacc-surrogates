#!/bin/bash

#SBATCH -J run-epidemic-training               # Job name
#SBATCH -o log.%j                         	 # Name of stdout output file (%j expands to jobId)
#SBATCH -p gh-dev                       # Queue name
#SBATCH -N 1                                 # Total number of nodes requested (56 cores/node)
#SBATCH -n 1                                 # Total number of mpi tasks requested
#SBATCH -t 2:00:00                          # Run time (hh:mm:ss)
#SBATCH -A XXXXXXXX							 # Project charge code

# Load CUDA module
module load cuda/12.8

# Load tacc-surrogates libraries
source /scratch/10386/lsmith9003/py-envs/tacc-surrogates/bin/activate
export PYTHONPATH=$SCRATCH/scripts/tacc-surrogates

# Run flow bench training/evaluation
python -W ignore main.py
