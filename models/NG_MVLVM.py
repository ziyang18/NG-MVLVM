from models_utility.function_gp import lt_log_determinant
from torch import triangular_solve
from sklearn.decomposition import PCA
import numpy as np
import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
from torch.distributions import kl_divergence
from torch.nn import functional as F
import gpytorch

torch.set_default_tensor_type(torch.DoubleTensor)
# torch.set_default_tensor_type(torch.FloatTensor)

zitter = 1e-8


class NG_MVLVM(nn.Module):
    """
    Next-Gen Multi-View Latent Variable Model (NG-MVLVM).
    
    Implements the NG-MVLVM with NG-SM (Next-Gen Spectral Mixture) kernel
    using random Fourier features approximation.
    """
    def __init__(self, num_batch, num_sample_pt, param_dict, Y, device=None, ifPCA=True):
        super(NG_MVLVM, self).__init__()
        self.device = device
        self.name = None
        self.num_view = len(Y)
        self.num_batch = num_batch
        self.num_samplept = num_sample_pt  # L/2: number of spectral points per mixture component
        self.latent_dim = param_dict['latent_dim']  # Q: latent dimension
        self.N = param_dict['N']  # N: number of observations
        self.num_m = param_dict['num_m']  # M: list of number of mixture components per view
        self.noise = param_dict['noise_err']
        self.lr_hyp = param_dict['lr_hyp']

        Y_tep= []
        Y_dim = []
        for i in range(self.num_view):
            Y_dim.append(Y[i].shape[1])
            Y_tep.append(torch.tensor(Y[i],device=self.device))
        self.Y_dim = Y_dim #list
        self.Y = Y_tep

        total_num_sample = []
        for i in range(self.num_view):
            total_num_sample.append(self.num_samplept * self.num_m[i])
        self.total_num_sample = total_num_sample #list
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        # NG-SM kernel parameters: weights (alpha), correlation (rho), means (mu1, mu2)
        log_weight = []
        rho = []
        mu1 = []
        mu2 = []
        for i in range(self.num_view):
            log_weight.append(nn.Parameter(torch.randn(self.num_m[i], 1, device=self.device), requires_grad=True))
            rho.append(nn.Parameter(torch.zeros(self.num_m[i], device=self.device), requires_grad=True))
            # shape: M * 1
            if self.num_m[i] == 1:
                # If SE kernel is used, then mu = 0, and requires_grad=False
                mu1.append(nn.Parameter(torch.zeros(self.num_m[i], self.latent_dim, device=self.device),
                                       requires_grad=False))
                mu2.append(nn.Parameter(torch.zeros(self.num_m[i], self.latent_dim, device=self.device),
                                       requires_grad=False))
            else:
                mu1.append(nn.Parameter(torch.zeros(self.num_m[i], self.latent_dim, device=self.device),
                                       requires_grad=True))
                mu2.append(nn.Parameter(torch.zeros(self.num_m[i], self.latent_dim, device=self.device),
                                       requires_grad=True))

        self.log_weight = nn.ParameterList(log_weight)
        self.rho = nn.ParameterList(rho)
        self.mu1 = nn.ParameterList(mu1)
        self.mu2 = nn.ParameterList(mu2)

        log_std1 = []
        log_std2 = []
        for i in range(self.num_view):
            log_std1.append(nn.Parameter(torch.ones(self.num_m[i], self.latent_dim, device=self.device)))
            log_std2.append(nn.Parameter(torch.ones(self.num_m[i], self.latent_dim, device=self.device)))
        self.log_std1 = nn.ParameterList(log_std1)
        self.log_std2 = nn.ParameterList(log_std2)


        if ifPCA:
            pca = PCA(n_components=self.latent_dim)
            X = pca.fit_transform(self.Y[0].cpu())
        else:
            X = torch.randn(self.N, self.latent_dim, device=self.device)

        # Latent variable X: variational posterior q(X) = N(mu_x, sigma_x^2)
        self.mu_x = nn.Parameter(torch.tensor(X, device=self.device), requires_grad=True)  # shape: N * Q
        self.log_sigma_x = nn.Parameter(torch.zeros(self.N, self.latent_dim, device=self.device), requires_grad=True)

    def _compute_sm_basis(self, x_star=None, f_eval=False, view=0):
        """
        Compute random Fourier features (RFF) basis for NG-SM kernel.
        
        Parameters
        ----------
        x_star : torch.Tensor, optional
            Prediction input points
        f_eval : bool
            If True, use mean of q(X) for evaluation; otherwise sample from q(X)
        view : int
            View index
        
        Returns
        -------
        torch.Tensor or tuple
            RFF basis Phi, and optionally Phi_star for predictions
        """
        multiple_Phi = []
        current_sampled_spectral_list1 = []
        current_sampled_spectral_list2 = []

        if f_eval:  # Use mean of q(X) to evaluate the latent function f
            x = self.mu_x
        else:
            # Sample from variational posterior q(X)
            std = F.softplus(self.log_sigma_x)  # shape: N * Q
            eps = torch.randn_like(std)
            x = self.mu_x + eps * std
        
        # NG-SM kernel parameters
        SM_weight = F.softplus(self.log_weight[view])  # alpha: M*1
        SM_std1 = F.softplus(self.log_std1[view])  # M * Q
        SM_std2 = F.softplus(self.log_std2[view])  # M * Q
        rho = self.rho[view]
        
        # Two-step reparameterization trick for bivariate Gaussian mixture
        for i_th in range(self.num_m[view]):
            SM_eps1 = torch.randn(self.num_samplept, self.latent_dim, device=self.device)  # L/2 * Q
            SM_eps2 = torch.randn(self.num_samplept, self.latent_dim, device=self.device)
            sampled_spectral_pt1 = self.mu1[view][i_th] + SM_std1[i_th] * SM_eps1
            sampled_spectral_pt2 = self.mu2[view][i_th] + rho[i_th] * (SM_std2[i_th] / SM_std1[i_th]) * (
                        sampled_spectral_pt1 - self.mu1[view][i_th]) + torch.sqrt(1 - rho[i_th] ** 2) * SM_std1[i_th] * SM_eps2

            if x_star is not None:
              current_sampled_spectral_list1.append(sampled_spectral_pt1)
              current_sampled_spectral_list2.append(sampled_spectral_pt2)

            # Compute random Fourier features
            x_spectral1 = (2 * np.pi) * x.matmul(sampled_spectral_pt1.t())  # N * L/2
            x_spectral2 = (2 * np.pi) * x.matmul(sampled_spectral_pt2.t())  # N * L/2

            Phi_i_th = (SM_weight[i_th] / (4 * self.num_samplept)).sqrt() * torch.cat(
                [x_spectral1.cos() + x_spectral2.cos(), x_spectral1.sin() + x_spectral1.sin()], 1)

            multiple_Phi.append(Phi_i_th)

        if x_star is None:
            return torch.cat(multiple_Phi, 1)  # N * (M * L)
        else:
            # Compute Phi_star for prediction points
            multiple_Phi_star = []
            for i_th, current_sampled in enumerate(current_sampled_spectral_list1):
                current_sampled1 = current_sampled
                current_sampled2 = current_sampled_spectral_list2[i_th]
                xstar_spectral1 = (2 * np.pi) * x_star.matmul(current_sampled1.t())  # N_star * L/2
                xstar_spectral2 = (2 * np.pi) * x_star.matmul(current_sampled2.t())  # N_star * L/2
                Phistar_i_th = (SM_weight[i_th] / (4 * self.num_samplept)).sqrt() * torch.cat(
                    [xstar_spectral1.cos() + xstar_spectral2.cos(), xstar_spectral1.sin() + xstar_spectral1.sin()], 1)
                multiple_Phi_star.append(Phistar_i_th)
            return torch.cat(multiple_Phi, 1), torch.cat(multiple_Phi_star, 1)  # N * (M * L), N_star * (M * L)


    def _compute_gram_approximate(self, Phi):  # shape:  (m*L) x (m*L)
        return Phi.t() @ Phi + (self.likelihood.noise + zitter).expand(Phi.shape[1], Phi.shape[1]).diag().diag()


    def _compute_gram_approximate_2(self, Phi):  # shape:  N x N
        return Phi @ Phi.T


    def _kl_div_qp(self):

        # shape: N x Q
        q_dist = torch.distributions.Normal(loc=self.mu_x, scale= F.softplus(self.log_sigma_x))
        p_dist = torch.distributions.Normal(loc=torch.zeros_like(q_dist.loc), scale=torch.ones_like(q_dist.loc))

        return kl_divergence(q_dist, p_dist).sum().div(self.N * self.latent_dim)

    def compute_lossv(self, batch_y, kl_option, view=0):
        """
        Compute negative log-likelihood for a single view.
        
        Parameters
        ----------
        batch_y : torch.Tensor
            Observations for the current view, shape: N * obs_dim
        kl_option : bool
            Whether to include KL divergence term
        view : int
            View index
        
        Returns
        -------
        torch.Tensor
            Negative log-likelihood (approximate lower bound)
        """
        obs_dim = batch_y.shape[1]
        obs_num = batch_y.shape[0]
        batch_y = torch.tensor(batch_y, device=self.device, dtype=torch.double)
        Phi = self._compute_sm_basis(view = view)

        # Negative log-marginal likelihood
        if Phi.shape[0] > Phi.shape[1]:  # If N > (M * L), use Woodbury identity
            Approximate_gram = self._compute_gram_approximate(Phi)  # shape: (M * L) x (M * L)
            L = torch.cholesky(Approximate_gram)
            Lt_inv_Phi_y = triangular_solve((Phi.t()).matmul(batch_y), L, upper=False)[0]

            neg_log_likelihood = (0.5 / self.likelihood.noise) * (batch_y.pow(2).sum() - Lt_inv_Phi_y.pow(2).sum())
            neg_log_likelihood += lt_log_determinant(L)
            neg_log_likelihood += (-self.total_num_sample[view]) * 2 * self.likelihood.noise.sqrt()
            neg_log_likelihood += 0.5 * obs_num * (np.log(2 * np.pi) + 2 * self.likelihood.noise.sqrt())

        else:
            # Direct computation when N <= (M * L)
            k_matrix = self._compute_gram_approximate_2(Phi=Phi)  # shape: N x N
            C_matrix = k_matrix + self.likelihood.noise * torch.eye(self.N, device=self.device)
            L = torch.cholesky(C_matrix)  # shape: N x N
            L_inv_y = triangular_solve(batch_y, L, upper=False)[0]

            # Compute log-likelihood
            constant_term = 0.5 * obs_num * np.log(2 * np.pi) * obs_dim
            log_det_term = torch.diagonal(L, dim1=-2, dim2=-1).sum().log() * obs_dim
            yy = 0.5 * L_inv_y.pow(2).sum()
            neg_log_likelihood = (constant_term + log_det_term + yy).div(obs_dim * obs_num)
        return neg_log_likelihood


    def compute_loss(self, batch_y, kl_option):
        """
        Compute total ELBO loss for multi-view data.
        
        Parameters
        ----------
        batch_y : list
            List of observations for each view
        kl_option : bool
            Whether to include KL divergence term
        
        Returns
        -------
        torch.Tensor
            Total ELBO loss (negative)
        """
        kl_x = self._kl_div_qp().div(self.N * 50)
        total_loss = 0.0
        for i in range(self.num_view):
            total_loss = total_loss + self.compute_lossv(batch_y[i], kl_option, view=i)

        total_loss = total_loss + kl_x
        return total_loss

    def f_eval(self, batch_y_view, x_star=None, view=0):
        """
        Evaluate the latent mapping function f.
        
        Parameters
        ----------
        batch_y_view : torch.Tensor
            Observations for characterizing the GP, shape: N * obs_dim
        x_star : torch.Tensor, optional
            Prediction input points, shape: N_star * Q
        view : int
            View index
        
        Returns
        -------
        tuple
            (f_star, k_matrix): predicted function values and kernel matrix
        """
        batch_y = torch.tensor(batch_y_view, device=self.device, dtype=torch.double)

        if x_star is None:
            x_star = self.mu_x

        Phi, Phi_star = self._compute_sm_basis(x_star=x_star, f_eval=True, view=view)

        cross_matrix = Phi_star @ Phi.T  # shape: N_star * N

        k_matrix = self._compute_gram_approximate_2(Phi=Phi)  # shape: N * N
        C_matrix = k_matrix + self.likelihood.noise * torch.eye(self.N, device=self.device)

        L = torch.cholesky(C_matrix)  # shape: N x N
        L_inv_y = triangular_solve(batch_y, L, upper=False)[0]  # inv(L) * y
        K_L_inv = triangular_solve(cross_matrix.T, L, upper=False)[0]  # inv(L) * K_{N, N_star}

        f_star = K_L_inv.T @ L_inv_y  # shape: N_star * obs_dim
        return f_star, k_matrix

