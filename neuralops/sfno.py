import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
from neuralop.models import SFNO
from omegaconf import DictConfig, OmegaConf
import hydra
import pdb
from neuralops.neuralop_helpers import *

# TODO: checkpointing
# TODO: gridpoint-specific abs_diff metric, diff metric for every group?
# TODO: W&B

# https://github.com/omry/omegaconf/issues/392#issuecomment-948956968
OmegaConf.register_new_resolver("as_tuple", lambda *args: tuple(args))

@hydra.main(version_base=None, config_path="../conf/neuralops/sfno", config_name="config")
def main(cfg : DictConfig):
    cfg = cfg.configs
    pl.seed_everything(cfg.env.seed)
    sfno = SFNO(**cfg.sfno.hyperparams)
    model = NeuralOpWrapper(sfno, cfg.sfno.optim, cfg.sfno.scheduler)
    trainer = pl.Trainer(**cfg.trainer)
    train_loader, val_loader = get_era5_datasets(**cfg.era5)
    trainer.fit(model=model, train_dataloaders=train_loader)

class NeuralOpWrapper(pl.LightningModule):
    def __init__(self, model, optimizer_cfg, scheduler_cfg):
        super().__init__()
        self.model = model
        self.loss_fn = HookedLpLoss(d=2)
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch['x']
        y_hat = self.model(x)
        y = batch['y']
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        print(f'train_loss: {loss}')
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), **self.hparams.optimizer_cfg)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **self.hparams.scheduler_cfg)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }

if __name__ == "__main__":
    main()