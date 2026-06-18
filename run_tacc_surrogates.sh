#!/bin/bash

#SBATCH -J run-epidemic-training               # Job name
#SBATCH -o log.%j                         	 # Name of stdout output file (%j expands to jobId)
#SBATCH -p gh-dev                       # Queue name
#SBATCH -N 2                                 # Total number of nodes requested (56 cores/node)
#SBATCH -n 2                                 # Total number of mpi tasks requested
#SBATCH -t 2:00:00                          # Run time (hh:mm:ss)
#SBATCH -A XXXXXXXX							 # Project charge code

# Load CUDA module
module load cuda/12.8

# Load tacc-surrogates libraries
source /scratch/10386/lsmith9003/py-envs/tacc-surrogates/bin/activate
export PYTHONPATH=$SCRATCH/scripts/tacc-surrogates

# Run training/evaluation (single node)
#python -W ignore main.py

# Run training/evaluation (multinode)
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
mpirun -np $SLURM_NTASKS --map-by ppr:1:node run_tacc_surrogates_idev.sh $MASTER_ADDR $SLURM_JOB_NUM_NODES
