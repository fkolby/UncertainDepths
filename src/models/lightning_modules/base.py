import pytorch_lightning as pl
import torch
from torch import optim, nn
import wandb
import pdb

from src.utility.viz_utils import log_images, log_loss_metrics


class Base_module(pl.LightningModule):
    def __init__(self, model, loss_function, steps_per_epoch: int, cfg, use_full_size_loss=False):
        super().__init__()
        self.model = model
        self.loss_function = loss_function
        self.lr = cfg.hyperparameters.learning_rate
        self.tstep = 0
        self.min_depth = cfg.dataset_params.min_depth
        self.max_depth = cfg.dataset_params.max_depth
        self.input_height = cfg.dataset_params.input_height
        self.input_width = cfg.dataset_params.input_width
        self.batch_size = cfg.hyperparameters.batch_size
        self.epochs = cfg.trainer_args.max_epochs
        self.steps_per_epoch = steps_per_epoch
        try:  # set it if it is in struct
            self.use_full_size_loss = cfg.dataset_params.use_full_size_loss
        except:
            self.use_full_size_loss = False

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        x, y = batch

        try:
            assert (x[:, 0, :, :].shape == y[:, 0, :, :].shape) & (
                x[0, 0, :, :].shape == torch.Size((self.input_height, self.input_width))
            )
        except:
            pdb.set_trace()

        preds = self(x)

        print(f"TRAIN:  x: {x.shape} y: {y.shape}, pred: {preds.shape}, tstep: {self.tstep}")

        if self.tstep % 1000 == 0 or (
            self.tstep < 1000 and self.tstep % 100 == 0
        ):  # dont log every single image (space issues. (space issues.)
            log_images(
                img=x[0, :, :, :].detach(),
                depth=y[0, :, :, :].detach(),
                pred=preds[0, :, :, :].detach(),
                vmin=self.min_depth,
                vmax=self.max_depth,
                step=self.tstep,
            )
        mask = torch.logical_and(
            y >= self.min_depth, y <= self.max_depth
        )  # perhaps also punish above maxdepth during training?
        loss = self.loss_function(preds[mask], y[mask])

        self.log("train_loss", loss)
        wandb.log(
            {"train_loss": loss, "learning_rate": self.lr_schedulers().get_last_lr()[0]},
            step=self.tstep,
        )
        # Log also full-size version (what we eventually will be evaluated on)

        log_loss_metrics(
            preds=preds[mask].detach(),
            targets=y[mask].detach(),
            tstep=self.tstep,
            loss_prefix="train_fullsize",
        )
        self.tstep += 1
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        assert (x[:, 0, :, :].shape == y[:, 0, :, :].shape) & (
            x[0, 0, :, :].shape == torch.Size((self.input_height, self.input_width))
        )
        preds = self(x)
        print(f"VALIDATION: x: {x.shape} y: {y.shape}, pred: {preds.shape}")

        mask = torch.logical_and(y >= self.min_depth, y <= self.max_depth)
        loss = self.loss_function(preds[mask], y[mask])

        wandb.log({"val_loss": loss}, step=self.tstep)
        self.log("validation_loss", loss)

        log_loss_metrics(
            preds=preds[mask].detach(),
            targets=y[mask].detach(),
            tstep=self.tstep,
            loss_prefix="val",
        )
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr, epochs=self.epochs, steps_per_epoch=self.steps_per_epoch
        )
        opt_dict = {
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            "optimizer": optimizer,
        }
        return opt_dict
