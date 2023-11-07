import torch
import torch.nn as nn
import design_bench

from torch.distributions.normal import Normal
from model.vae import Encoder, Decoder, Prior
from utils import TASK_ABBREVIATIONS, DISCRETE

class ExPT(nn.Module):
    def __init__(
        self,
        task_name: str,
        d_model=128,
        mlp_ratio=4,
        nhead=4,
        dropout=0.1,
        activation='gelu',
        norm_first=False,
        num_layers=4,
        dim_z=32,
        depth_vae=4,
        hidden_vae=512,
        vae_norm_first=True,
        learn_prior=False,
        beta_vae=1.0,
        std_decoder=0.5
    ):
        if task_name != 'tf10':
            task = design_bench.make(TASK_ABBREVIATIONS[task_name])
        else:
            task = design_bench.make(TASK_ABBREVIATIONS[task_name], dataset_kwargs={"max_samples": 10000})
        if task_name in DISCRETE:
            task.map_to_logits()
            x = task.x
            x = x.reshape(x.shape[0], -1)
            dim_x = x.shape[-1]
        else:
            dim_x = task.x.shape[-1]
        self.dim_x = dim_x
        self.task_name = task_name

        self.ctx_emb = nn.Linear(dim_x + 1, d_model) # embed (x, y) context pairs
        self.y_tar_emb = nn.Linear(1, d_model) # embed y target

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model*mlp_ratio, dropout,
            activation, batch_first=True, norm_first=norm_first
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.learn_prior = learn_prior
        self.dim_z = dim_z
        self.beta_vae = beta_vae
        
        # vae model on top to model x | y, x_C, y_C
        self.vae_encoder = Encoder(self.dim_x, dim_z, hidden_vae, depth_vae, norm_first=vae_norm_first, embed_y=False, skip_dim=d_model)
        self.vae_decoder = Decoder(self.dim_x, dim_z, hidden_vae, depth_vae, norm_first=vae_norm_first, embed_y=False, skip_dim=d_model, std=std_decoder)
        if learn_prior:
            self.vae_prior = Prior(dim_z, hidden_vae, norm_first=vae_norm_first, embed_y=False, inp_dim=d_model)
            
    def create_mask(self, num_ctx, num_tar, device):
        num_all = num_ctx + num_tar

        mask = torch.zeros(num_all, num_all, device=device)
        mask[:, num_ctx:] = float('-inf')

        return mask, num_tar

    def encode(self, xc, yc, yt):
        # xc: B, C, dim_x
        # yc: B, C, 1
        # yt: B, T, 1

        ctx_emb = self.ctx_emb(torch.cat((xc, yc), dim=-1)) # B, C, D
        tar_y_emb = self.y_tar_emb(yt) # B, T, D

        inputs = torch.cat((ctx_emb, tar_y_emb), dim=1)
        tnp_mask, num_tar = self.create_mask(ctx_emb.shape[1], tar_y_emb.shape[1], device=ctx_emb.device)
        tnp_mask = tnp_mask.to(inputs.dtype)
        out_transformers = self.encoder(inputs, mask=tnp_mask) # B, C + T, D

        return out_transformers[:, -num_tar:]
    
    def forward_vae(self, ht: torch.Tensor, xt: torch.Tensor, n_samples, reduce_mean):
        # ht: B, T, d_model
        # xt: B, T, dim_x
        
        ht = ht.flatten(0, 1) # BxT, d_model
        
        # posterior and samples
        q_z: Normal = self.vae_encoder(xt.flatten(0, 1), ht) # q(z | x, h)
        z_sample = q_z.rsample((n_samples,)).transpose(0, 1) # BxT, n_samples, dim_z
        
        # decode
        out_transformer_rep = ht.unsqueeze(1).repeat_interleave(n_samples, dim=1) # BxT, n_samples, d_model
        p_x: Normal = self.vae_decoder(z_sample, out_transformer_rep) # p(x | z, h)
        
        # prior
        if self.learn_prior:
            p_z = self.vae_prior(ht)
        else:
            p_z = Normal(
                torch.zeros((out_transformer_rep.shape[0], self.dim_z)).to(z_sample.device),
                torch.ones((out_transformer_rep.shape[0], self.dim_z)).to(z_sample.device)
            )
        
        x_rep = xt.flatten(0, 1).unsqueeze(1).repeat(1, n_samples, 1) # BxT, n_samples, dim_x
        ll = p_x.log_prob(x_rep).mean(dim=(-2, -1))
        l2 = torch.mean((x_rep - p_x.loc)**2, dim=(-2, -1))
        kl = torch.distributions.kl_divergence(q_z, p_z).mean(dim=-1)
        if reduce_mean:
            ll = torch.mean(ll)
            l2 = torch.mean(l2)
            kl = torch.mean(kl)
        
        return ll, l2, kl
    
    def forward(self, xc, yc, xt, yt, n_samples=10):
        ht = self.encode(xc, yc, yt)
        ll, l2, kl = self.forward_vae(ht, xt, n_samples, reduce_mean=True)
        loss = -ll + self.beta_vae * kl
        loss_dict = {'nll_x': -ll, 'l2_x': l2, 'kl_x': kl, 'loss': loss}
        return loss, loss_dict

    def sample(self, xc, yc, yt, n_samples):
        # xc: B, C, dim_x
        # yc: B, C, 1
        # yt: B, T, 1
        b, t, _ = yt.shape
        ht = self.encode(xc, yc, yt) # B, T, D
        ht = ht.unsqueeze(0).repeat(n_samples, 1, 1, 1).flatten(0, 2) # n_samples x B x T, D
        # ht = ht.flatten(0, 1) # BxT, d_model
        
        # sample from prior
        if self.learn_prior:
            p_z = self.vae_prior(ht)
        else:
            p_z = Normal(
                torch.zeros((ht.shape[0], self.dim_z)).to(ht.device),
                torch.ones((ht.shape[0], self.dim_z)).to(ht.device)
            )
        z_sample = p_z.rsample()

        x_sample = self.vae_decoder(z_sample, ht)

        return x_sample.loc.unflatten(0, sizes=(n_samples, b, t)).squeeze(2).transpose(0, 1)
