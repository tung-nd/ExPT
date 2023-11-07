import numpy as np
import torch
import gpytorch

from typing import Dict
from torch.utils.data import IterableDataset
from data.gp import StandardGP, SpectralMixtureGP


def get_range(x: np.ndarray):
    x_range = []
    x_dim = x.shape[-1]
    for i in range(x_dim):
        x_ = x[:, i]
        min_dim, max_dim = x_.min(), x_.max()
        x_range.append((min_dim, max_dim))
    return x_range


### equal ratios for all gp kernels for now
class MultipleGPIterableDataset(IterableDataset):
    def __init__(
        self,
        x_real: torch.Tensor,
        normalize_x_scheme: bool,
        batch_size: int = 32,
        gp_type: Dict = {},
        gp_args: Dict = {},
        kernel_use_flag = '1',
        num_ctx: int = 1024,
        num_tar: int = 128,
        noise_x: float = 0.0,
        device: str = 'cuda',
    ):
        super().__init__()
        
        self.x_real = x_real.to(device)
        self.dim_x = self.x_real.shape[1]

        # individual batch size is batch_size // n_kernels except for the last kernel
        n_kernels = np.sum(kernel_use_flag)
        gp_batch_size = batch_size // n_kernels
        list_batch_size = [batch_size // n_kernels for _ in range(n_kernels - 1)]
        list_batch_size.append(batch_size - gp_batch_size * (n_kernels - 1))

        # self.gp is a dictionary of all kernels used for generating
        self.gp = {}
        if len(self.x_real.shape) == 2:
            dim_gp = self.dim_x
        else:
            dim_gp = self.dim_x * self.x_real.shape[2]
        gp_type = {k: gp_type[k] for i, k in enumerate(gp_type.keys()) if kernel_use_flag[i]}
        gp_args = {k: gp_args[k] for i, k in enumerate(gp_args.keys()) if kernel_use_flag[i]}
        for i, kernel in enumerate(gp_args.keys()):
            if gp_type[kernel] == 'standard':
                gp_class = StandardGP
            elif gp_type[kernel] == 'mixture':
                gp_class = SpectralMixtureGP
            else:
                raise NotImplementedError

            self.gp[kernel] = gp_class(
                x_dim=dim_gp,
                batch_size=list_batch_size[i],
                gp_args=gp_args[kernel]
            ).to(device)
            self.gp[kernel].eval()

        self.normalize_x_scheme = normalize_x_scheme
        self.list_batch_size = list_batch_size
        self.total_batch_size = batch_size
        self.num_ctx = num_ctx
        self.num_tar = num_tar
        self.noise_x = noise_x
        self.device = device
    
    def __iter__(self):
        num_points = self.num_ctx + self.num_tar
        while True:
            # sample data from self.x
            xs = []
            for _ in range(self.total_batch_size):
                # if number of real data points > num_points, sample without replacement
                # else sample with replacement
                if self.x_real.shape[0] > num_points:
                    rand_ids = torch.randperm(self.x_real.shape[0])[:num_points]
                else:
                    rand_ids = torch.randint(0, self.x_real.shape[0], (num_points,))
                xs.append(self.x_real[rand_ids])
            xs = torch.stack(xs, dim=0)
            noise = torch.randn_like(xs) * self.noise_x
            xs = xs + noise
            if self.normalize_x_scheme == 'scale':
                xs = torch.clamp(xs, min=0.0, max=1.0)
            
            # gen data from each kernel
            ys = []
            start_idx = 0
            with torch.no_grad():
                for i, kernel in enumerate(self.gp.keys()):
                    batch_size = self.list_batch_size[i]
                    end_idx = start_idx + batch_size
                    x = xs[start_idx:end_idx]
                    if len(x.shape) == 4: # discrete tasks
                        x = x.flatten(-2, -1)
                    gp = self.gp[kernel]
                    with gpytorch.settings.prior_mode(True):
                        y = gp(x, random_parameter=True).rsample().unsqueeze(-1)
                    
                    ys.append(y)
                    start_idx += batch_size
                
                ys = torch.cat(ys, dim=0)

            xc, yc = xs[:, :self.num_ctx], ys[:, :self.num_ctx]
            xt, yt = xs[:, self.num_ctx:], ys[:, self.num_ctx:]

            yield xc, yc, xt, yt


class IndividualIterableDataset(IterableDataset):
    def __init__(self, dataset: MultipleGPIterableDataset):
        super().__init__()
        self.dataset = dataset

    def __iter__(self):
        for xc, yc, xt, yt in self.dataset:
            for i in range(xc.shape[0]):
                yield xc[i], yc[i], xt[i], yt[i]
