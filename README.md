TACC-Surrogates is a series of python wrappers and environment configuration files intended to streamline the training and evaluation of surrogate models on TACC systems. In this repository, we collect a number of popular surrogate model architectures, and outfit each with a common initialization, training, and evaluation method.

As an example, training a Fourier Neural Operator (FNO) is as simple as intializing the model's hyperparameters:

```python
print('initializing FNO')
fno_test = FNO(
        n_modes=(16,16),
        hidden_channels=32,
        in_channels=(2+3*num_prior_timestep),
        out_channels=(3*num_forward_timestep),
        n_epoch=10,
        batch_size=16
        )
print('done')
```
And calling the class's built-in training function:

```python
print('starting training')
fno_test.train(data_in,data_out)
print('done')
```
Evaluation (i.e., forward prediction of unseen data) is called just as easily:

```python
print('Evaluating...')
x0 = data_in[0:2]
data_pred = fno_test.eval(x0)
```

The FNO architecture can be swapped with any member of the 'models' sub-directory. The goal is to allow users to quickly swap model architectures without having to reformat their data.

The architectures in this repository are valid for any training dataset cast into "input/output" format. Time series data is formatted with prior time steps as input, and future time steps as output. test_fno.py provides an example of how to format time series data for training, using the [FlowBench dataset](https://huggingface.co/datasets/BGLab/FlowBench) as an example. run_tacc_surrogates.sh provides an example of how to submit a tacc-surrogates training job to TACC's Lonestar6.
