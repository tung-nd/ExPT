import os

from data.gp_dataset import MultipleGPIterableDataset
from pytorch_lightning.cli import LightningCLI
from data.gp_datamodule import GPDataModule
from model.expt_module import ExPTModule
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

os.environ['NCCL_P2P_DISABLE']='1'
# torch.set_float32_matmul_precision('medium')

def main():
    # Initialize Lightning with the model and data modules, and instruct it to parse the config yml
    cli = LightningCLI(
        model_class=ExPTModule,
        datamodule_class=GPDataModule,
        seed_everything_default=42,
        save_config_overwrite=True,
        run=False,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )
    os.makedirs(cli.trainer.default_root_dir, exist_ok=True)

    logger_name = cli.trainer.logger.name
    cli.trainer.logger = TensorBoardLogger(
        save_dir=cli.trainer.default_root_dir,
        name=f'{logger_name}/logs',
        version=None,
        log_graph=False,
        default_hp_metric=True,
        prefix=""
    )

    for i in range(len(cli.trainer.callbacks)):
        if isinstance(cli.trainer.callbacks[i], ModelCheckpoint):
            cli.trainer.callbacks[i] = ModelCheckpoint(
                dirpath=os.path.join(cli.trainer.default_root_dir, logger_name, 'checkpoints'),
                save_last=True,
                every_n_train_steps=cli.trainer.callbacks[i]._every_n_train_steps,
                verbose=False,
                filename="epoch_{epoch:03d}",
                auto_insert_metric_name=False
            )

    # fit() runs the training
    cli.trainer.validate(cli.model, datamodule=cli.datamodule)


if __name__ == "__main__":
    main()
