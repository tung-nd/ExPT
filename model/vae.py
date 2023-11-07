import torch
import torch.nn as nn
import design_bench

from torch.distributions.normal import Normal
from utils import TASK_ABBREVIATIONS

def build_blocks(dim_in, hidden_dim, dim_out, depth, norm_first, embed, skip_dim):
    skip_connection_dim = hidden_dim if embed else skip_dim
    if norm_first:
        blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_in+skip_connection_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )
        ])
    else:
        blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_in+skip_connection_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
            )
        ])

    for _ in range(depth - 2):
        if norm_first:
            blocks.append(
                nn.Sequential(
                    nn.Linear(hidden_dim+skip_connection_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                )
            )
        else:
            blocks.append(
                nn.Sequential(
                    nn.Linear(hidden_dim+skip_connection_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                )
            )
    
    blocks.append(nn.Linear(hidden_dim+skip_connection_dim, dim_out*2))

    return blocks

class Encoder(nn.Module):
    def __init__(self, dim_x, dim_z, hidden_dim, depth, norm_first, embed_y, skip_dim=None):
        super().__init__()
        
        if embed_y:
            self.y_embed = nn.Linear(1, hidden_dim)
        else:
            self.y_embed = nn.Identity()

        skip_dim = 1 if skip_dim is None else skip_dim

        self.blocks = build_blocks(dim_x, hidden_dim, dim_z, depth, norm_first, embed_y, skip_dim)
                    
    def forward(self, x, y):
        # return q(z | x, y)
        y_embed = self.y_embed(y)

        out = x
        for block in self.blocks:
            out = block(torch.cat((out, y_embed), dim=-1))
        
        mu, std = torch.chunk(out, 2, dim=-1)
        std = torch.exp(std)
        return Normal(mu, std)
        
class Decoder(nn.Module):
    def __init__(self, dim_x, dim_z, hidden_dim, depth, norm_first, embed_y, skip_dim=None, std=1.0):
        super().__init__()
        
        if embed_y:
            self.y_embed = nn.Linear(1, hidden_dim)
        else:
            self.y_embed = nn.Identity()

        skip_dim = 1 if skip_dim is None else skip_dim

        self.blocks = build_blocks(dim_z, hidden_dim, dim_x, depth, norm_first, embed_y, skip_dim)
        self.std = std
            
    def forward(self, z, y):
        # return p(x | z, y)
        y_embed = self.y_embed(y)
        
        out = z 
        for block in self.blocks:
            out = block(torch.cat((out, y_embed), dim=-1))
        
        mu, _ = torch.chunk(out, 2, dim=-1)
        # std = torch.exp(std)
        std = torch.ones_like(mu) * self.std
        return Normal(mu, std)

class Prior(nn.Module):
    def __init__(self, dim_z, hidden_dim, norm_first, embed_y, inp_dim=1):
        super().__init__()
        
        if embed_y:
            self.y_embed = nn.Linear(1, hidden_dim)
            inp_dim = hidden_dim
        else:
            self.y_embed = nn.Identity()

        if norm_first:
            self.block_0 = nn.Sequential(
                nn.Linear(inp_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )
            
            self.block_1 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )
        else:
            self.block_0 = nn.Sequential(
                nn.Linear(inp_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
            )
            
            self.block_1 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
            )
        
        self.final = nn.Linear(hidden_dim, dim_z*2)
            
    def forward(self, y):
        # return p(z | y)
        y_embed = self.y_embed(y)
        
        z = self.block_0(y_embed)
        z = self.block_1(z)
        z = self.final(z)
        
        mu, std = torch.chunk(z, 2, dim=-1)
        std = torch.exp(std)
        return Normal(mu, std)

class VAE(nn.Module):
    def __init__(self, task_name, hidden_dim, depth, dim_z, init_method, embed_y=True, learn_prior=False):
        super().__init__()

        task = design_bench.make(TASK_ABBREVIATIONS[task_name])
        dim_x = task.x.shape[-1]
        
        self.task_name = task_name
        self.dim_x = dim_x
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.dim_z = dim_z
        self.embed_y = embed_y
        self.learn_prior = learn_prior
        
        self.encoder = Encoder(dim_x, dim_z, hidden_dim, depth, embed_y)
        self.decoder = Decoder(dim_x, dim_z, hidden_dim, depth, embed_y)

        if learn_prior:
            self.prior = Prior(dim_z, hidden_dim, embed_y)

        if init_method == 'uniform':
            self.apply(self._init_weights_uniform)
        elif init_method == 'normal':
            self.apply(self._init_weights_normal)
        elif init_method == 'xavier_uniform':
            self.apply(self._init_weights_xavier_uniform)
        elif init_method == 'xavier_normal':
            self.apply(self._init_weights_xavier_normal)
        elif init_method == 'kaiming_uniform':
            self.apply(self._init_weights_kaiming_uniform)
        elif init_method == 'kaiming_normal':
            self.apply(self._init_weights_kaiming_normal)
        elif init_method == 'orthogonal':
            self.apply(self._init_weights_orthogonal)
        else:
            raise NotImplementedError()

    def _init_weights_uniform(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.uniform_(m.weight, a=-0.1, b=0.1)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)

    def _init_weights_normal(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.1)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)

    def _init_weights_xavier_uniform(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)

    def _init_weights_xavier_normal(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)

    def _init_weights_kaiming_uniform(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)

    def _init_weights_kaiming_normal(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)
    
    def _init_weights_orthogonal(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)
        
    def encode(self, x, y):
        return self.encoder(x, y)
    
    def decode(self, z, y):
        return self.decoder(z, y)

    def sample_prior(self, y, n_samples):
        if self.learn_prior:
            p_z = self.prior(y)
        else:
            p_z = Normal(
                torch.zeros((y.shape[0], self.dim_z)).to(y.device),
                torch.ones((y.shape[0], self.dim_z)).to(y.device)
            )
        return p_z.rsample((n_samples,)).transpose(0, 1)
    
    def predict(self, y, n_samples):
        z_sample = self.sample_prior(y, n_samples) # B, n_samples, D
        y = y.unsqueeze(1).repeat(1, n_samples, 1)
        p_x = self.decode(z_sample, y)
        return p_x.loc
        
    def forward(self, x: torch.Tensor, y: torch.Tensor, n_samples=10):
        if self.learn_prior:
            p_z = self.prior(y)
        else:
            p_z = Normal(
                torch.zeros((x.shape[0], self.dim_z)).to(y.device),
                torch.ones((x.shape[0], self.dim_z)).to(y.device)
            )
            
        q_z: Normal = self.encode(x, y)
        z_sample = q_z.rsample((n_samples,)).transpose(0, 1) # B, n_samples, D
        y_rep = y.unsqueeze(1).repeat_interleave(n_samples, dim=1)
        p_x: Normal = self.decode(z_sample, y_rep)
        x_rep = x.unsqueeze(1).repeat_interleave(n_samples, dim=1)
        
        ll = p_x.log_prob(x_rep).mean()
        l2 = torch.mean((x_rep - p_x.loc)**2)
        kl = torch.distributions.kl_divergence(q_z, p_z).mean()
        
        return ll, l2, kl
    
# model = VAE(56, 128, 4, 16, 'uniform')
# x = torch.randn(16, 56)
# y = torch.randn(16, 1)
# ll, kl = model(x, y)
# print (ll)
# print (kl)