# TACC-Surrogates

TACC-Surrogates is a series of python wrappers and environment configuration files intended to streamline the training and evaluation of surrogate models on TACC systems. In this repository, we collect a number of popular surrogate model architectures, and outfit each with a common initialization, training, and evaluation method.

The main advantage of TACC-Surrogates is its common data structure. Our intent is for the user
to freely swap model architectures without having to reformat their dataset. Toward this end, we adopt the following standard shape for input/output datasets:

(batch, num_timesteps_prior, **features)

Here, batch corresponds to the batch dimension, num_timesteps_prior corresponds to the number
of previous timesteps available to a given forward prediction (the same idea as sequence length),
and **features corresponds to the data features associated with each timestep. 
These features can be organizaed as a single, flattened array, or as 
a multi-dimensional grid (if one chooses an architecture that exploits spatial correlations). 

Each member of the ```models``` subdirectory is configured to accept datasets in the standard form. 


## Installing on Lonestar6

Installing tacc-surrogates on the Lonestar6 cluster is as simple as cloning this repository:
```
git clone https://github.com/lsmithTACC/tacc-surrogates.git
```

And executing a requirements install with pip:
```
cd tacc-surrogates
pip install tacc-surrogates/requirements.txt
```

If you encounter any issues with the requirements file, or if you want to offload environment maintenance, we are also hosting a pre-installed version of the environment on a TACC staff account. You can activate it with the following commands (permissions are set such that any user can activate):
```
module load cuda/12.2
source /scratch/10386/lsmith9003/python-envs/tacc-surrogates/bin/activate
```

## Training on Existing Architectures

The user can train on any architecture in the ```models``` sub-directory with just a few function calls. Namely, if the user has already loaded the correct envionrment and shaped their data according to the standard format, they need only (1) initialize an instance of the model architecture, and (2) call the built-in .train method. 

As an example, if one wishes to train their dataset using the Fourier Neural Operator (FNO), 
they need only run the following commands:

```
# 1) Initialize FNO
fno_test = FNO(
        n_modes=(16,16),
        hidden_channels=32,
        n_epoch=2,
        batch_size=16,
        num_prior=num_timesteps_prior,
        num_forward=num_timesteps_forward,
        num_vector_components=data_in.shape[-1]
        )

# 2) Train
fno_test.train(data_in,data_out)
```

The arguments for initialization (n_epoch, batch_size, etc.) can be found by looking at the relevant file within the ```models``` sub-directory. Most initialization steps are quite similar.

If there is any confusion regarding the initialization/training of a given architecture, the ```tests``` sub-directory contains a complete submission script for certain models, with the [FlowBench dataset](https://baskargroup.bitbucket.io/) serving as a test bed. This sub-directory will be updated as tests are completed on new architectures.


## How to Add Your Own Architecture to the Models Directory

Adding a new architecure to TACC-Surrogates consists of the following two steps:

1) Adding a new entry to the ```models``` sub-directory
2) Adding a demonstration/verification of the model to the ```tests``` sub-directory

Each entry in the ```models``` sub-directory inherits the Base_Model class defined in Base.py. This script contains the default training and evaluation methods. When writing a new model file, one will generally need to append the inherited base class with:

1) An __init__ method, which defines the input arguments to your new model.
2) A model object, which defines/links to your custom architecture. The model object is defined as self.model and must be a torch.nn module object. Note that self.model does not need to be configured to accept the standard data format as input (we typically leave this as a separate data packing step).
3) A loss-function object, which defines the loss function for your training loop. This is defined as self.loss_function and again must be a torch.nn module object.
4) A data-packing method. This method is repsonsible for converting input data from the standard format to a format accepted by self.model. Some model calls (such as Pytorch's LSTM) already work with the standard format, while others (such as the FNO) require shifting data channels.

We recommend the FNO.py script as a reliable template for constructing new model architecture files.


## List of Architectures to be Added

The following is a master list of all architectures slated to be added to the ```models``` sub-directory. An (x) will be placed next to architectures that have been successfully uploaded and tested.

- DMD
- Neural ODE 
- FNO (x)
- LSTM 
- MLP
- GINO - Geometry informed neural operator
- U-Net
- Deep-O-Net
- PINN
- GNN
- SINDy 
- Hamiltonian/Lagrangian NN
- PINO 
