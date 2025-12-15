"""
Multi-view s-curve comparison demo for NG-MVLVM.

This script demonstrates the NG-MVLVM (Next-Gen Multi-View Latent Variable Model)
on synthetic s-curve data with two views.
"""

import sys
import matplotlib.pylab as plt
import torch
from tqdm import tqdm
import os
import numpy as np
from numpy.random import RandomState
import random

import os
import sys
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_mine import load_dataset
from models.NG_MVLVM import NG_MVLVM
from visualizer import Visualizer
from metrics import (knn_classify,
                     mean_squared_error,
                     r_squared)


def save_models(model, optimizer, epoch, losses, result_dir, data_name, save_model=True):
    """
    Save model checkpoint.
    
    Parameters
    ----------
    model : nn.Module
        Model to save
    optimizer : torch.optim.Optimizer
        Optimizer state
    epoch : int
        Current epoch
    losses : float
        Current loss value
    result_dir : str
        Result saving path
    data_name : str
        Data name
    save_model : bool
        Whether to save the model
    """
    state = {'model': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'epoch': epoch,
             'losses': losses}
    if save_model:
        log_dir = result_dir + f"{data_name}_epoch{epoch}.pt"
        torch.save(state, log_dir)


# Hyperparameters (as specified in paper appendix)
random_seed = 8


def reset_seed(seed: int) -> None:
    """Reset random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


reset_seed(random_seed)
device = 'cpu'  # or 'cuda' if GPU available

# Load Dataset
rng = RandomState(random_seed)

# Load two views of s-curve data
ds2 = load_dataset(rng, 's-curve2', 'gaussian')
Y2 = ds2.Y / np.linalg.norm(ds2.Y)

ds1 = load_dataset(rng, 's-curve', 'gaussian')
Y1 = ds1.Y / np.linalg.norm(ds1.Y)

# Hyperparameter settings (as per paper appendix)
# Following the notation in the paper:
# - M: number of NG-SM mixture components per view
# - L: number of random features per mixture component (L/2 spectral points per component)
# - Q: latent dimension
# - N: number of observations
# - learning_rate: Adam optimizer learning rate
# - noise_variance: initial observation noise variance (learned during training)
# - num_iterations: number of training iterations
# - kl_weight: weight for KL divergence term (1/(N*50) as in paper)

hyperparams = {}
hyperparams['M'] = [2, 2]  # M: Number of NG-SM mixture components per view (paper notation)
hyperparams['L'] = 50  # L: Number of random features per mixture component (L/2 spectral points)
hyperparams['Q'] = ds1.latent_dim  # Q: Latent dimension (typically 2 for s-curve)
hyperparams['N'] = ds1.Y.shape[0]  # N: Number of observations (500 for s-curve)
hyperparams['learning_rate'] = 0.01  # Adam optimizer learning rate
hyperparams['noise_variance'] = 100.0  # Initial noise variance (learned via GaussianLikelihood)
hyperparams['num_iterations'] = 10000  # Number of training iterations
hyperparams['kl_weight'] = 1.0 / (hyperparams['N'] * 50)  # KL divergence weight: 1/(N*50)
hyperparams['use_kl'] = True  # Whether to include KL regularization term

# Prepare multi-view data
Y = [Y1, Y2]

# Initialize NG-MVLVM model
# Note: The internal parameter names (num_m, num_sample_pt, etc.) are kept for compatibility
# with the model implementation, but they correspond to paper notation:
# - num_m -> M (mixture components)
# - num_sample_pt -> L (random features per component)
# - latent_dim -> Q (latent dimension)
# - noise_err -> initial noise variance
# - lr_hyp -> learning rate
model = NG_MVLVM(
    num_batch=1,
    num_sample_pt=hyperparams['L'],  # L: random features per mixture component
    param_dict={
        'num_m': hyperparams['M'],  # M: number of mixture components per view
        'num_sample_pt': hyperparams['L'],  # L: random features per component
        'latent_dim': hyperparams['Q'],  # Q: latent dimension
        'N': hyperparams['N'],  # N: number of observations
        'noise_err': hyperparams['noise_variance'],  # Initial noise variance
        'lr_hyp': hyperparams['learning_rate']  # Learning rate
    },
    Y=Y,
    device=device,
    ifPCA=True
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])

# Training loop
epochs_iter = tqdm(range(hyperparams['num_iterations'] + 1), desc="Epoch")
for i in epochs_iter:
    model.train()
    optimizer.zero_grad()
    loss_total = model.compute_loss(batch_y=Y, kl_option=hyperparams['use_kl'])
    loss_total.backward()
    optimizer.step()

    if i % 500 == 0:
        print(f'\nELBO: {loss_total.item()}')
        print(f"X_KL: {model._kl_div_qp().item()}")

        # Evaluate and visualize view 1
        F, K = model.f_eval(batch_y_view=Y1, view=0, x_star=None)
        F = (F - 5).cpu().detach().numpy()  # Data generation adds mean=5
        K = K.cpu().detach().numpy()
        model_name = f"NG_MVLVM_Y1_M{hyperparams['M'][0]}_L{hyperparams['L']}"
        res_dir = f'./results/{model_name}/'
        os.makedirs(res_dir, exist_ok=True)
        viz = Visualizer(res_dir + 'figures', ds1)
        viz.plot_iteration(i + 1, Y=0, F=0, K=K, X=model.mu_x.cpu().detach().numpy())

        # Evaluate and visualize view 2
        F, K = model.f_eval(batch_y_view=Y2, view=1, x_star=None)
        F = (F - 5).cpu().detach().numpy()  # Data generation adds mean=5
        K = K.cpu().detach().numpy()
        model_name = f"NG_MVLVM_Y2_M{hyperparams['M'][1]}_L{hyperparams['L']}"
        res_dir = f'./results/{model_name}/'
        os.makedirs(res_dir, exist_ok=True)
        viz = Visualizer(res_dir + 'figures', ds2)
        viz.plot_iteration(i + 1, Y=0, F=0, K=K, X=model.mu_x.cpu().detach().numpy())

        save_models(model=model, optimizer=optimizer, epoch=i, losses=loss_total,
                    result_dir=res_dir, data_name='s-curve2', save_model=False)

        # Compute R^2 score if ground truth X is available
        if ds1.has_true_X:
            r2_X = r_squared(model.mu_x.cpu().detach().numpy(), ds1.X)
            print(f'R^2 X: {r2_X}')

        print("\n")
