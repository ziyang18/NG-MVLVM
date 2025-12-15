 NG-MVLVM: Next-Gen Multi-View Latent Variable Model

This repository contains the implementation of **NG-MVLVM** (Next-Gen Multi-View Latent Variable Model) as described in the paper "Multi-View Oriented GPLVM: Expressiveness and Efficiency".

## Overview

NG-MVLVM is a multi-view Gaussian process latent variable model that uses the **NG-SM (Next-Gen Spectral Mixture) kernel** with random Fourier features approximation for scalable variational inference. The model learns unified latent representations from multi-view data while maintaining computational efficiency.


## 📦 Setup / Installation

### Install dependencies

We recommend using a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```


## 🧪 Running the Demo

### Multi-View S-Curve Demo

This demo demonstrates NG-MVLVM on synthetic s-curve data with two views:

```bash
python demo/MV_s_shape_comparisions.py
```

The demo will:
- Load synthetic s-curve data (two views)
- Train NG-MVLVM with hyperparameters as specified in the paper appendix
- Visualize learned latent representations
- Compute R² score for latent space recovery

## 📝 Citation

If you use this code, please cite:

```bibtex
@inproceedings{yang2025multi,
  title     = {Multi-View Oriented {GPLVM}: Expressiveness and Efficiency},
  author    = {Yang, Zi and Li, Ying and Lin, Zhidi and Zhang, Michael Minyi and Olmos, Pablo M.},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025}
}
```


## 🔗 References

- Paper: Multi-View Oriented GPLVM: Expressiveness and Efficiency
- The NG-SM kernel is based on modeling spectral density with bivariate Gaussian mixtures
- Random Fourier features enable scalable variational inference
