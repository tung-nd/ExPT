import torch
import design_bench
import numpy as np

from typing import Dict, Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from pytorch_lightning.trainer.supporters import CombinedLoader
from data.gp_dataset import MultipleGPIterableDataset, IndividualIterableDataset, get_range
from data.eval_dataset import RealEvaluationDataset
from data import prepare_training_xs, prepare_eval_data
from utils import TASK_ABBREVIATIONS, NAME_TO_FULL_DATASET, DISCRETE


class GPDataModule(LightningDataModule):
    def __init__(
        self,
        task: str,
        normalize_x_scheme: str,
        normalize_y_scheme: str,
        gp_type: Dict = {},
        gp_args: Dict = {},
        kernel_use_flag: str = '1',
        num_ctx: int = 100,
        num_tar: int = 128,
        noise_x: float = 0.1,
        device: str = 'cuda',
        batch_size: int = 128,
        inner_batch_size: int = 128,
        pin_memory: bool = False,
        eval_kernels: list = ['dkitty'],
        eval_data_ratio: float = 1.0,
        eval_samping_strategy: str = 'random',
        eval_batch_size: int = 1,
    ):
        """
        task: abbreviation of the target domain we are pretraining the model for, e.g., dkitty
        normalize_x_scheme: either scale (to [0, 1]), standardize (to N(0, 1)), or none (no normalization)
        normalize_y_scheme: same as x
        gp_type: a dictionary where keys are the kernel names and values are either 'standard' or 'mixture', only use 'standard' for now
        gp_args: a dictionary where keys are the kernel names and values are their hypeparameter range
        kernel_use_flag: a string of 0s and 1s, 1 means use the corresponding kernel, e.g., '101' means use the first and third kernels
        noise_x: std of gaussian noise added to x
        device: device to generate data from GP
        batch_size: batch size used to train the model
        inner_batch_size: batch size used to generate data from GP
        eval_kernels: names of evaluation kernels, can be synthetic or real
        eval_data_ratio: ratio of real (x,y) pairs used for evaluation
        eval_samping_strategy: random or poor
        eval_batch_size: batch size used to evaluate the model
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        self.hparams.kernel_use_flag = [int(i) for i in kernel_use_flag]
        if np.sum(self.hparams.kernel_use_flag) == 0:
            raise "Must use at least 1 kernel"

        self.data_train: Optional[MultipleGPIterableDataset] = None
        self.data_val: Optional[Dict] = None
    
    def denormalize_x(self, x, kernel):
        if isinstance(self.mean_x_eval[kernel], torch.Tensor):
            mean_x, std_x = self.mean_x_eval[kernel].to(x.device), self.std_x_eval[kernel].to(x.device)
        else:
            mean_x, std_x = self.mean_x_eval[kernel], self.std_x_eval[kernel]
        return x * std_x + mean_x
    
    def denormalize_y(self, y, kernel):
        mean_y, std_y = self.mean_y_eval[kernel].to(y.device), self.std_y_eval[kernel].to(y.device)
        return y * std_y + mean_y

    def normalize_y(self, y, kernel):
        mean_y, std_y = self.mean_y_eval[kernel].to(y.device), self.std_y_eval[kernel].to(y.device)
        return (y - mean_y) / std_y

    def setup(self, stage: Optional[str] = None):
        # load datasets only if they're not loaded already
        x_train, mean_x, std_x = prepare_training_xs(
            task_name=self.hparams.task,
            normalize_x_scheme=self.hparams.normalize_x_scheme,
        )
        x_train = (x_train - mean_x) / std_x
        self.mean_x_train = mean_x
        self.std_x_train = std_x
        
        if not self.data_train:
            self.data_train = IndividualIterableDataset(
                MultipleGPIterableDataset(
                    x_real=x_train,
                    normalize_x_scheme=self.hparams.normalize_x_scheme,
                    batch_size=self.hparams.inner_batch_size,
                    gp_type=self.hparams.gp_type,
                    gp_args=self.hparams.gp_args,
                    kernel_use_flag=self.hparams.kernel_use_flag,
                    num_ctx=self.hparams.num_ctx,
                    num_tar=self.hparams.num_tar,
                    noise_x=self.hparams.noise_x,
                    device=self.hparams.device,
                )
            )
        
        if not self.data_val:
            self.data_val = {}
            self.mean_x_eval = {}
            self.std_x_eval = {}
            self.mean_y_eval = {}
            self.std_y_eval = {}
            for kernel in self.hparams.eval_kernels:
                x_eval, mean_x_eval, std_x_eval, y_eval, mean_y_eval, std_y_eval = prepare_eval_data(
                    kernel,
                    normalize_x_scheme=self.hparams.normalize_x_scheme,
                    normalize_y_scheme=self.hparams.normalize_y_scheme,
                )
                x_eval = (x_eval - mean_x_eval) / std_x_eval
                y_eval = (y_eval - mean_y_eval) / std_y_eval
                self.mean_x_eval[kernel] = mean_x_eval
                self.std_x_eval[kernel] = std_x_eval
                self.mean_y_eval[kernel] = mean_y_eval
                self.std_y_eval[kernel] = std_y_eval

                if self.hparams.eval_data_ratio > 1:
                    len_eval = int(self.hparams.eval_data_ratio)
                    if self.hparams.eval_samping_strategy == 'random':
                        eval_ids = torch.arange(len_eval)
                    else:
                        assert self.hparams.eval_samping_strategy == 'poor'
                        eval_ids = torch.argsort(y_eval.squeeze(-1))[:len_eval]
                else:
                    len_eval = int(self.hparams.eval_data_ratio * x_eval.shape[0])
                    if self.hparams.eval_samping_strategy == 'random':
                        eval_ids = torch.arange(len_eval)
                    else:
                        assert self.hparams.eval_samping_strategy == 'poor'
                        eval_ids = torch.argsort(y_eval.squeeze(-1))[:len_eval]

                x_public = x_eval[eval_ids]
                y_public = y_eval[eval_ids]

                if kernel != 'tf10':
                    hidden_dataset = NAME_TO_FULL_DATASET[TASK_ABBREVIATIONS[kernel]]()
                else:
                    hidden_dataset = NAME_TO_FULL_DATASET[TASK_ABBREVIATIONS[kernel]](max_samples=50000)
                x_hidden, y_hidden = hidden_dataset.x, hidden_dataset.y
                
                if kernel in DISCRETE:
                    if kernel != 'tf10':
                        task = design_bench.make(TASK_ABBREVIATIONS[kernel])
                    else:
                        task = design_bench.make(TASK_ABBREVIATIONS[kernel], dataset_kwargs={"max_samples": 10000})
                    x_hidden = task.to_logits(x_hidden).reshape(x_hidden.shape[0], -1)
                
                x_hidden = torch.from_numpy(x_hidden)
                y_hidden = torch.from_numpy(y_hidden)
                x_hidden = (x_hidden - mean_x_eval) / std_x_eval
                y_hidden = (y_hidden - mean_y_eval) / std_y_eval

                self.data_val[kernel] = RealEvaluationDataset(
                    public_x=x_public.unsqueeze(0),
                    public_y=y_public.unsqueeze(0),
                    hidden_x=x_hidden.unsqueeze(0),
                    hidden_y=y_hidden.unsqueeze(0),
                )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=0,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self):
        loaders = {
            kernel: DataLoader(
                dataset,
                batch_size=self.hparams.eval_batch_size,
                num_workers=0,
                pin_memory=self.hparams.pin_memory,
            ) for kernel, dataset in self.data_val.items()
        }
        combined_loaders = CombinedLoader(loaders, mode="max_size_cycle")
        return combined_loaders
