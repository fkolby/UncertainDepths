import pdb
from typing import Union

import pytorch_lightning as pl
import torch
from torch import nn, optim

import wandb
from src.utility.viz_utils import log_images, log_loss_metrics
from src.models.laplace.online_laplace import OnlineLaplace


class Base_lightning_module(pl.LightningModule):
    def __init__(
        self,
        model,
        loss_function,
        steps_per_epoch: int,
        cfg,
        dataset_size: Union[None, int] = None,
        use_full_size_loss=False,
    ):
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
        self.cfg = cfg
        self.steps_per_epoch = steps_per_epoch
        try:  # set it if it is in struct
            self.use_full_size_loss = cfg.dataset_params.use_full_size_loss
        except:
            self.use_full_size_loss = False

        if cfg.models.model_type == "Online_Laplace":
            assert not (dataset_size is None)
            self.Online_Laplace = OnlineLaplace(
                net=self.model.stochastic_net,
                dataset_size=dataset_size,
                loss_function=loss_function,
                cfg=cfg,
                device="cuda",
            )

    def forward(self, inputs):
        if self.cfg.models.model_type == "Online_Laplace":
            print("dont use forward for online laplace - use step or similar")
            raise (ModuleNotFoundError)
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch  # third argument is only used during eval (OOD-classification)

        try:
            assert (x[:, 0, :, :].shape == y[:, 0, :, :].shape) & (
                x[0, 0, :, :].shape == torch.Size((self.input_height, self.input_width))
            )
        except:
            pdb.set_trace()

        mask = torch.logical_and(
            y >= self.min_depth, y <= self.max_depth
        )  # perhaps also punish above maxdepth during training?

        if self.cfg.models.model_type == "Online_Laplace":
            if self.cfg.models.override_constant_hessian_memory_factor:
                out_dict = self.Online_Laplace.step(
                    img=x,
                    depth=y,
                    train=True,
                    hessian_memory_factor=1 - self.lr_schedulers().get_last_lr()[0],
                )
            else:
                out_dict = self.Online_Laplace.step(img=x, depth=y, train=True)
            loss = out_dict["loss"]
            preds = out_dict["preds"]
            variance = out_dict["variance"]
            wandb.log(
                {
                    "Time spent on forward pass": out_dict["time_forward"],
                    "Time spent on hessian calculation": out_dict["time_hessian"],
                    "Time spent in rest of step": out_dict["time_rest"],
                    "Time spent in total of step": out_dict["time_total"],
                    "Time spent in tough calculation of step": out_dict["time_tough"],
                    "Time spent in second_hess group": out_dict["time_second_hess"],
                    "Time spent in appending": out_dict["time_append"],
                    "Size of change in hessian": out_dict["size_of_change"],
                    "Absolute mean change in hessian": out_dict["abs_size_of_change"],
                    "Current mean hessian values": out_dict["hessian_size"],
                    "Median of hessian: ": out_dict["hessian_median"],
                    "10th quantile of hessian": out_dict["hessian_tenth_qt"],
                    "90th quantile of hessian": out_dict["hessian_ninetieth_qt"],
                },
                step=self.tstep,
            )
        else:
            preds = self(x)

            loss = self.loss_function(preds[mask], y[mask])

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
        x, y, _ = batch  # third argument is only used during eval (OOD-classification)

        assert (x[:, 0, :, :].shape == y[:, 0, :, :].shape) & (
            x[0, 0, :, :].shape == torch.Size((self.input_height, self.input_width))
        )

        mask = torch.logical_and(
            y >= self.min_depth, y <= self.max_depth
        )  # perhaps also punish above maxdepth during training?

        if self.cfg.models.model_type == "Online_Laplace":
            out_dict = self.Online_Laplace.step(img=x, depth=y, train=False)
            loss = out_dict["loss"]
            preds = out_dict["preds"]
            variance = out_dict["variance"]
        else:
            preds = self(x)

            loss = self.loss_function(preds[mask], y[mask])

        print(f"VALIDATION: x: {x.shape} y: {y.shape}, pred: {preds.shape}")

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
