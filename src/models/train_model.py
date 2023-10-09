import logging
import argparse
import PIL
import seaborn as sns
import torch
import wandb
from torchvision import transforms
from lightning.pytorch import loggers
import hydra
from omegaconf import OmegaConf, DictConfig
from src.models.modelImplementations.baseUNet import BaseUNet
from src.data.datamodules import KITTIDataModule
import pytorch_lightning as pl
from torch import optim, nn
from pytorch_lightning import callbacks
import pdb
from src.utility.viz_utils import log_images, log_loss_metrics
import os


class KITTI_depth_lightning_module(pl.LightningModule):
    def __init__(
        self,
        model,
        loss_function,
        learning_rate,
        min_depth,
        max_depth,
        input_height,
        input_width,
        batch_size,
    ):
        super().__init__()
        self.model = model
        self.loss_function = loss_function
        self.lr = learning_rate
        self.tstep = 0
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.input_height = input_height
        self.input_width = input_width
        self.batch_size = batch_size

    def training_step(self, batch, batch_idx):
        x, y, fullsize_targets = batch
        try:
            assert (x[:, 0, :, :].shape == y[:, 0, :, :].shape) & (
                x[0, 0, :, :].shape == torch.Size((self.input_height, self.input_width))
            )

        except:
            pdb.set_trace()
        preds = self.model(x)
        print(f"TRAIN:  x: {x.shape} y: {y.shape}, pred: {preds.shape}, tstep: {self.tstep}")
        if self.tstep % 10 == 0:
            log_images(
                img=torch.squeeze(x[0, :, :, :].detach(), dim=0),
                depth=torch.squeeze(y[0, :, :, :].detach(), dim=0),
                pred=torch.squeeze(preds[0, :, :, :].detach(), dim=0),
                vmin=self.min_depth,
                vmax=self.max_depth,
                step=self.tstep,
            )
        mask = torch.logical_and(
            y > self.min_depth, y < self.max_depth
        )  # perhaps also punish above maxdepth during training?
        loss = self.loss_function(preds * mask, y * mask)

        self.log("train_loss", loss)
        wandb.log({"train_loss": loss}, step=self.tstep)

        fullsize_mask = torch.logical_and(
            fullsize_targets > self.min_depth, fullsize_targets < self.max_depth
        )

        masked_full_size_targets = fullsize_targets * fullsize_mask

        resized_preds = nn.functional.interpolate(
            preds, fullsize_targets.shape[-2:], mode="bilinear", align_corners=True
        )

        masked_resized_preds = resized_preds * fullsize_mask
        log_loss_metrics(
            preds=masked_resized_preds.detach(),
            targets=masked_full_size_targets.detach(),
            tstep=self.tstep,
            loss_prefix="train",
        )
        self.tstep += 1
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, fullsize_targets = batch

        assert (x[:, 0, :, :].shape == y[:, 0, :, :].shape) & (
            x[0, 0, :, :].shape == torch.Size((self.input_height, self.input_width))
        )
        preds = self.model(x)
        print(f"VALIDATION: x: {x.shape} y: {y.shape}, pred: {preds.shape}")
        mask = torch.logical_and(y > self.min_depth, y < self.max_depth)
        loss = self.loss_function(preds * mask, y * mask)

        wandb.log({"val_loss": loss}, step=self.tstep)
        self.log("validation_loss", loss)

        fullsize_mask = torch.logical_and(
            fullsize_targets > self.min_depth, fullsize_targets < self.max_depth
        )

        masked_full_size_targets = fullsize_targets * fullsize_mask

        resized_preds = nn.functional.interpolate(
            preds, fullsize_targets.shape[-2:], mode="bilinear", align_corners=True
        )

        masked_resized_preds = resized_preds * fullsize_mask
        log_loss_metrics(
            preds=masked_resized_preds.detach(),
            targets=masked_full_size_targets.detach(),
            tstep=self.tstep,
            loss_prefix="val",
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        mask = torch.logical_and(y > self.min_depth, y < self.max_depth)
        loss = self.loss_function(preds * mask, y * mask)
        self.log("TEST loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    if cfg.in_debug:
        pdb.set_trace()
        os.environ["WANDB_MODE"] = "disabled"
        trainer_args = {
            "max_epochs": 1,
            "limit_val_batches": 0.001,
            "limit_train_batches": 0.001,
            "fast_dev_run": True,
        }
    else:
        os.environ["WANDB_MODE"] = "online"
        trainer_args = {"max_epochs": cfg.trainer_args.max_epochs}

    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logger = loggers.WandbLogger(project="UncertainDepths")
    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(cfg))
    model = KITTI_depth_lightning_module(
        model=BaseUNet(in_channels=3, out_channels=1),
        loss_function=nn.functional.mse_loss,
        learning_rate=cfg.hyperparameters.learning_rate,
        min_depth=cfg.dataset_params.min_depth,
        max_depth=cfg.dataset_params.max_depth,
        input_height=cfg.dataset_params.input_height,
        input_width=cfg.dataset_params.input_width,
        batch_size=cfg.hyperparameters.batch_size,
    )
    trainer = pl.Trainer(logger=logger, **trainer_args)
    transform = transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.CenterCrop((352, 1216)),
            transforms.Lambda(lambda x: x / 255),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # normalize using imagenet values, as I have yet not calced it for KITTI.
        ]
    )
    target_transform = transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.CenterCrop((352, 1216)),
            transforms.Lambda(lambda x: x / 256),  # 256 as per devkit
        ]
    )
    datamodule = KITTIDataModule(
        data_dir=cfg.dataset_params.data_dir,
        batch_size=cfg.hyperparameters.batch_size,
        transform=transform,
        target_transform=target_transform,
        num_workers=cfg.dataset_params.num_workers,
        input_height=cfg.dataset_params.input_height,
        input_width=cfg.dataset_params.input_width,
    )

    datamodule.setup(stage="fit")
    trainer.fit(
        model=model,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader(),
    )


if __name__ == "__main__":
    main()
