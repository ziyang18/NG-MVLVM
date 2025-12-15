"""============================================================================
Dataset loading functions for s-curve synthetic data.
============================================================================"""

import numpy as np
from .dataset import Dataset
from GPy import kern
from scipy.special import (expit as logistic,
                           logsumexp)
from sklearn.datasets import make_s_curve


def load_dataset(rng, name, emissions, test_split=0):
    """Load dataset by name.
    
    Parameters
    ----------
    rng : RandomState
        Random number generator
    name : str
        Dataset name ('s-curve' or 's-curve2')
    emissions : str
        Emission type ('gaussian', 'bernoulli', 'poisson', etc.)
    test_split : float
        Test split ratio (default: 0)
    
    Returns
    -------
    Dataset
        Dataset object
    """
    loader = {
        's-curve': gen_s_curve,
        's-curve2': gen_s_curve2,
    }[name]
    
    return loader(rng, emissions, test_split)


def gen_s_curve(rng, emissions, test_split):
    """Generate synthetic s-curve data.
    
    Parameters
    ----------
    rng : RandomState
        Random number generator
    emissions : str
        Emission type
    test_split : float
        Test split ratio
    
    Returns
    -------
    Dataset
        Dataset object
    """
    N = 500
    J = 100
    D = 2

    # Generate latent manifold
    X, t = make_s_curve(N, random_state=rng)
    X = np.delete(X, obj=1, axis=1)
    X = X / np.std(X, axis=0)
    inds = t.argsort()
    X = X[inds]
    t = t[inds]

    # Generate kernel K and latent GP-distributed maps F
    K = kern.RBF(input_dim=D, lengthscale=1).K(X)
    F = rng.multivariate_normal(np.zeros(N), K, size=J).T

    # Generate emissions using F and/or K
    if emissions == 'bernoulli':
        P = logistic(F)
        Y = rng.binomial(1, P).astype(np.double)
        return Dataset(rng, 's-curve', False, Y=Y, X=X, F=F, latent_dim=D,
                       labels=t, test_split=test_split)
    if emissions == 'gaussian':
        Y = F + np.random.normal(5, scale=0.5, size=F.shape)
        return Dataset(rng, 's-curve', False, Y=Y, X=X, F=F, K=K, latent_dim=D,
                       labels=t, test_split=test_split)
    elif emissions == 'multinomial':
        C = 100
        pi = np.exp(F - logsumexp(F, axis=1)[:, None])
        Y = np.zeros(pi.shape)
        for n in range(N):
            Y[n] = rng.multinomial(C, pi[n])
        return Dataset(rng, 's-curve', False, Y=Y, X=X, F=F, latent_dim=D, labels=t,
                       test_split=test_split)
    elif emissions == 'negbinom':
        P = logistic(F)
        R = np.arange(1, J + 1, dtype=float)
        Y = rng.negative_binomial(R, 1 - P)
        return Dataset(rng, 's-curve', False, False, Y=Y, X=X, F=F, R=R,
                       latent_dim=D, labels=t, test_split=test_split)
    else:
        assert (emissions == 'poisson')
        theta = np.exp(F)
        Y = rng.poisson(theta)
        return Dataset(rng, 's-curve', False, Y=Y, X=X, F=F, latent_dim=D,
                       labels=t, test_split=test_split)


def gen_s_curve2(rng, emissions, test_split):
    """Generate synthetic s-curve2 data (second view).
    
    Parameters
    ----------
    rng : RandomState
        Random number generator
    emissions : str
        Emission type
    test_split : float
        Test split ratio
    
    Returns
    -------
    Dataset
        Dataset object
    """
    N = 500
    J = 100
    D = 2

    # Generate latent manifold
    X, t = make_s_curve(N, random_state=rng)
    X = np.delete(X, obj=1, axis=1)
    X = X / np.std(X, axis=0)
    inds = t.argsort()
    X = X[inds]
    t = t[inds]

    # Generate kernel K and latent GP-distributed maps F
    kernel = kern.RBF(input_dim=D, lengthscale=1.0)
    K = kernel.K(X)
    F = rng.multivariate_normal(np.zeros(N), K, size=J).T

    # Generate emissions using F and/or K
    if emissions == 'bernoulli':
        P = logistic(F)
        Y = rng.binomial(1, P).astype(np.double)
        return Dataset(rng, 's-curve2', False, Y=Y, X=X, F=F, latent_dim=D,
                       labels=t, test_split=test_split)
    if emissions == 'gaussian':
        Y = F + np.random.normal(5, scale=0.5, size=F.shape)
        return Dataset(rng, 's-curve2', False, Y=Y, X=X, F=F, K=K, latent_dim=D,
                       labels=t, test_split=test_split)
    elif emissions == 'multinomial':
        C = 100
        pi = np.exp(F - logsumexp(F, axis=1)[:, None])
        Y = np.zeros(pi.shape)
        for n in range(N):
            Y[n] = rng.multinomial(C, pi[n])
        return Dataset(rng, 's-curve2', False, Y=Y, X=X, F=F, latent_dim=D, labels=t,
                       test_split=test_split)
    elif emissions == 'negbinom':
        P = logistic(F)
        R = np.arange(1, J + 1, dtype=float)
        Y = rng.negative_binomial(R, 1 - P)
        return Dataset(rng, 's-curve2', False, False, Y=Y, X=X, F=F, R=R,
                       latent_dim=D, labels=t, test_split=test_split)
    else:
        assert (emissions == 'poisson')
        theta = np.exp(F)
        Y = rng.poisson(theta)
        return Dataset(rng, 's-curve2', False, Y=Y, X=X, F=F, latent_dim=D,
                       labels=t, test_split=test_split)
