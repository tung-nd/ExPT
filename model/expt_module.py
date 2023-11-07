
import torch
import design_bench

from typing import Any
from pytorch_lightning import LightningModule
from model.expt import ExPT
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils import DISCRETE, TASK_ABBREVIATIONS, REAL

class ExPTModule(LightningModule):
    def __init__(
        self,
        net: ExPT,
        lr: float = 5e-4,
        beta_1: float = 0.9,
        beta_2: float = 0.95,
        weight_decay: float = 1e-5,
        warmup_iters: int = 1000,
        max_iters: int = 10000,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-8,
        eval_pred: bool = True,
        eval_optimize: bool = True,
        condition_type: int = 0,
        n_samples: int = 256,
        pretrained_path: str = "",
    ):
        """
        net: the model
        beta_1, beta_2: betas in Adam optimizer
        warmup_iters, max_iters, warmup_start_lr, eta_min: HPs of lr scheduler
        eval_pred: if True perform predictive evaluation
        eval_optimize: if True perform optimization evaluation
        condition_type: either 0 or 1, if 0 use the best y in the dataset, if 1 use the y* in the hidden dataset
        n_samples: the number of samples drawn from the model
        """
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        if len(pretrained_path) > 0:
            self.load_pretrained_weights(pretrained_path)

    def load_pretrained_weights(self, pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location=torch.device("cpu"))
        print("Loading pre-trained checkpoint from: %s" % pretrained_path)
        checkpoint_model = checkpoint["state_dict"]

        state_dict = self.state_dict()
        for k in list(checkpoint_model.keys()):
            if k not in state_dict.keys() or checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # load pre-trained model
        msg = self.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    def training_step(self, batch: Any, batch_idx: int):
        xc, yc, xt, yt = batch

        loss, loss_dict = self.net.forward(xc, yc, xt, yt)

        for var in loss_dict.keys():
            self.log(
                "train/" + var,
                loss_dict[var],
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
        
        return loss

    def eval_pred(self, kernel, batch, prefix):
        xc, yc, xt, yt = batch

        # only evaluate on out-of-distribution data
        # assume there's only one batch (true for real data)
        if kernel in REAL:
            y_ood = yt[yt > yc.max()]
            len_ood = len(y_ood)
            sort_ids = torch.argsort(yt, dim=1) # B, N, 1: from worst to best data points
            data_ids = sort_ids[:, -len_ood:]
            xt = xt[torch.arange(xt.shape[0]).unsqueeze(-1), data_ids.squeeze(-1)] # B, n, D
            yt = yt[torch.arange(yt.shape[0]).unsqueeze(-1), data_ids.squeeze(-1)] # B, n, 1

        max_tar_len = 1000 # only evaluate 1000 target points at once due to memmory constraint
        total_len = xt.shape[1]
        start_idx = 0
        cnt = 0
        loss_dict_total = {}
        while start_idx < total_len:
            end_idx = start_idx + min(max_tar_len, total_len - start_idx)
            _, loss_dict = self.net.forward(xc, yc, xt[:, start_idx:end_idx], yt[:, start_idx:end_idx])

            for k in loss_dict.keys():
                if k not in loss_dict_total:
                    loss_dict_total[k] = loss_dict[k]
                else:
                    loss_dict_total[k] += loss_dict[k]
            
            start_idx = end_idx
            cnt += 1
        
        loss_dict_total = {k: v / cnt for k, v in loss_dict_total.items()}
        
        for var in loss_dict_total.keys():
            self.log(
                f"{prefix}/{kernel}_{var}",
                loss_dict_total[var],
                prog_bar=True,
                on_epoch=True,
                sync_dist=True
            )


    def eval_optimize(self, kernel, batch, prefix):
        # xc, yc are from the public dataset, and xt and yt are from the hidden dataset
        # only use hidden dataset to get the optimal y*
        xc, yc, _, yt = batch

        # if 0 use the best y in the dataset, if 1 use the y* in the hidden dataset
        if self.hparams.condition_type == 0:
            y_eval = yc.max()
        else:
            assert self.hparams.condition_type == 1
            y_eval = yt.max()
        y_eval = y_eval.reshape(-1, 1, 1) # B, 1, 1

        y_eval_unorm = self.trainer.datamodule.denormalize_y(y_eval, kernel)
        self.log(
            f"{prefix}/{kernel}_conditioning_value",
            y_eval_unorm.mean(),
            prog_bar=True,
            on_epoch=True,
            sync_dist=True
        )
        
        best_y_dataset, _ = torch.max(yc, dim=1)
        best_y_dataset = self.trainer.datamodule.denormalize_y(best_y_dataset, kernel)
        best_y_dataset = best_y_dataset.mean()
        self.log(
            f"{prefix}/{kernel}_best_dataset",
            best_y_dataset,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True
        )

        xc = xc.to(self.device)
        yc = yc.to(self.device)
        y_eval = y_eval.to(self.device)
        y_eval_unorm = y_eval_unorm.to(self.device)

        # sample conditioning on y_eval
        xt_pred = self.net.sample(
            xc, yc, y_eval,
            n_samples=self.hparams.n_samples,
        ) # B, n_samples, dim_x
        
        xt_pred = xt_pred.flatten(0, 1) # B x n_samples, dim_x
        xt_pred = self.trainer.datamodule.denormalize_x(xt_pred, kernel)
        xt_pred = xt_pred.cpu().numpy()

        if kernel != 'tf10':
            oracle = design_bench.make(TASK_ABBREVIATIONS[kernel])
        else:
            oracle = design_bench.make(TASK_ABBREVIATIONS[kernel], dataset_kwargs={"max_samples": 10000})
        if kernel in DISCRETE:
            oracle.map_to_logits()
            xt_pred = xt_pred.reshape(xt_pred.shape[0], oracle.x.shape[1], oracle.x.shape[2])

        y_return = oracle.predict(xt_pred) # B x n_samples, 1            
        y_return = torch.from_numpy(y_return).unflatten(0, sizes=(1, -1)).to(y_eval_unorm.device, dtype=y_eval.dtype) # B, n_samples, 1
        
        # get the best point
        best_y_return = torch.max(y_return, dim=1)[0] # B, 1
        average_return = best_y_return.mean()

        self.log(
            f"{prefix}/{kernel}_max_return",
            average_return,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True
        )

        # get the 50th percentile point
        y_return_sorted, _ = torch.sort(y_return, dim=1)
        percentile_50_return = y_return_sorted[:, y_return_sorted.shape[1] // 2]
        percentile_50_return_avg = percentile_50_return.mean()

        self.log(
            f"{prefix}/{kernel}_median_return",
            percentile_50_return_avg,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True
        )

        self.log(
            f"{prefix}/{kernel}_mean_return",
            y_return.mean(),
            prog_bar=True,
            on_epoch=True,
            sync_dist=True
        )


    def validation_step(self, batch: Any, batch_idx: int):
        all_kernels = batch.keys()
        for kernel in all_kernels:
            if self.hparams.eval_pred:
                self.eval_pred(kernel, batch[kernel], prefix='val')
            if self.hparams.eval_optimize:
                self.eval_optimize(kernel, batch[kernel], prefix='val')

    def configure_optimizers(self):
        decay = []
        no_decay = []
        for name, m in self.named_parameters():
            if "pos_embed" in name:
                no_decay.append(m)
            else:
                decay.append(m)

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": decay,
                    "lr": self.hparams.lr,
                    "betas": (self.hparams.beta_1, self.hparams.beta_2),
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": no_decay,
                    "lr": self.hparams.lr,
                    "betas": (self.hparams.beta_1, self.hparams.beta_2),
                    "weight_decay": 0,
                },
            ]
        )

        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            self.hparams.warmup_iters,
            self.hparams.max_iters,
            self.hparams.warmup_start_lr,
            self.hparams.eta_min,
        )
        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
