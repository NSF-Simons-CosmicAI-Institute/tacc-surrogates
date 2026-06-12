# Load CUDA module
module load cuda/12.8

# Load tacc-surrogates libraries
source /scratch/10386/lsmith9003/py-envs/tacc-surrogates/bin/activate
export PYTHONPATH=$SCRATCH/scripts/tacc-surrogates:$PYTHONPATH

# Run flow bench training/evaluation
python -W ignore main.py
#python -W ignore model_analysis_test.py
#python -W ignore pca_analysis.py
