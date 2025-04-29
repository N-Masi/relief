import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from neuralop.models import SFNO
from omegaconf import DictConfig, OmegaConf
import hydra
import pdb
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuralops.neuralop_helpers import *

# TODO: gridpoint-specific abs_diff metric, diff metric for every group? Upload to W&B

# https://github.com/omry/omegaconf/issues/392#issuecomment-948956968
OmegaConf.register_new_resolver("as_tuple", lambda *args: tuple(args))

@hydra.main(version_base=None, config_path="../conf/neuralops/sfno", config_name="config")
def main(cfg : DictConfig):
    cfg = cfg.configs
    pl.seed_everything(cfg.env.seed)
    sfno = SFNO(**cfg.sfno.hyperparams)
    model = NeuralOpWrapper(sfno, cfg.sfno.optim, cfg.sfno.scheduler)
    train_loader, val_loader = get_era5_datasets(**cfg.era5)
    wandb_logger = WandbLogger(**cfg.wandb)
    trainer = pl.Trainer(
        **cfg.trainer, 
        default_root_dir=os.getcwd()+'/checkpoints/',
        logger=wandb_logger
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        ckpt_path="last"
    )

if __name__ == "__main__":
    main()