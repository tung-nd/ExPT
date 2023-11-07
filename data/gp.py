from typing import Dict
import gpytorch
import warnings

from gpytorch.kernels import ScaleKernel, SpectralMixtureKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.priors import UniformPrior
from utils import KERNEL_NAME_MAP


class BaseGP(ExactGP):
    def __init__(self, batch_size, gp_args: Dict):
        super().__init__(None, None, likelihood=GaussianLikelihood())

        self.batch_size = batch_size
        self.gp_args = gp_args

        self.mean_module = None
        self.covar_module = None
    
    def set_hyperparameters(self, hyperparameters: Dict):
        return

    def random_parameter(self, hyperparameters, device):
        return

    def forward(self, x, random_parameter=True, hyperparameters={}):
        device = x.device
        # Sample random hyperparameters
        if random_parameter:
            self.random_parameter(hyperparameters, device)
            self.set_hyperparameters(hyperparameters)
        
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def predict(self, x):
        likelihood = GaussianLikelihood().to(x.device)
        likelihood.eval()
        self.eval()
        return likelihood(self(x, random_parameter=False))


class StandardGP(BaseGP):
    def __init__(self, x_dim, batch_size, gp_args):
        super(StandardGP, self).__init__(batch_size, gp_args)

        self.mean_module = ConstantMean()

        self.prior = UniformPrior(gp_args['prior_low'], gp_args['prior_high'])
        self.scale_prior = UniformPrior(gp_args['scale_prior_low'], gp_args['scale_prior_high'])

        self.kernel = gp_args['kernel']
        kernel_args = {'batch_shape': (batch_size,)}
        self.ard_dim = None
        if self.kernel in ['rbf', 'matern', 'rq', 'piecewise']:
            kernel_args['ard_num_dims'] = x_dim
            self.ard_dim = kernel_args['ard_num_dims']
        if self.kernel == 'polynomial':
            kernel_args['power'] = 2

        self.covar_module = ScaleKernel(
            KERNEL_NAME_MAP[self.kernel](**kernel_args),
            batch_shape=(batch_size,)
        )

    def set_hyperparameters(self, hyperparameters: Dict):
        if self.kernel in ['cosine', 'periodic']:
            self.covar_module.base_kernel.period_length = hyperparameters['period_length']
            if self.kernel == 'periodic':
                self.covar_module.base_kernel.lengthscale = hyperparameters['lengthscale']
        elif self.kernel == 'linear':
            self.covar_module.base_kernel.variance = hyperparameters['variance']
        elif self.kernel == 'polynomial':
            self.covar_module.base_kernel.offset = hyperparameters['offset']
        elif self.kernel in ['rbf', 'matern', 'piecewise', 'rq']:
            self.covar_module.base_kernel.lengthscale = hyperparameters['lengthscale']
        else:
            pass
        self.covar_module.outputscale = hyperparameters['outputscale']
        return
    
    def random_parameter(self, hyperparameters, device):
        hyperparameters['period_length'] = self.prior.rsample((self.batch_size,)).to(device)
        if self.ard_dim is None:
            hyperparameters['lengthscale'] = self.prior.rsample((self.batch_size,)).to(device)
        else:
            hyperparameters['lengthscale'] = self.prior.rsample((self.batch_size, self.ard_dim)).to(device)
        hyperparameters['variance'] = self.prior.rsample((self.batch_size,)).to(device)
        hyperparameters['offset'] = self.prior.rsample((self.batch_size,)).to(device)
        hyperparameters['outputscale'] = self.scale_prior.rsample((self.batch_size,)).to(device)


class SpectralMixtureGP(BaseGP):
    def __init__(self, x_dim, batch_size, gp_args):
        super(SpectralMixtureGP, self).__init__(batch_size, gp_args)

        self.mean_module = ConstantMean()

        self.mean_prior = UniformPrior(gp_args['mean_prior_low'], gp_args['mean_prior_high'])
        self.scale_prior = UniformPrior(gp_args['scale_prior_low'], gp_args['scale_prior_high'])
        self.weight_prior = UniformPrior(gp_args['weight_prior_low'], gp_args['weight_prior_high'])
        self.num_mixtures = gp_args['num_mixtures']
        self.x_dim = x_dim

        self.covar_module = SpectralMixtureKernel(
            num_mixtures=self.num_mixtures,
            ard_num_dims=self.x_dim,
            # batch_shape=(batch_size,)
        )

    def set_hyperparameters(self, hyperparameters: Dict):
        self.covar_module.mixture_means = hyperparameters['mixture_means']
        self.covar_module.mixture_scales = hyperparameters['mixture_scales']
        self.covar_module.mixture_weights = hyperparameters['mixture_weights']
        return
    
    def random_parameter(self, hyperparameters, device):
        hyperparameters['mixture_means'] = self.mean_prior.rsample((self.num_mixtures, 1, self.x_dim)).to(device)
        hyperparameters['mixture_scales'] = self.scale_prior.rsample((self.num_mixtures, 1, self.x_dim)).to(device)
        weights = self.weight_prior.rsample((self.num_mixtures,)).to(device)
        hyperparameters['mixture_weights'] = weights / weights.sum(dim=-1, keepdim=True)