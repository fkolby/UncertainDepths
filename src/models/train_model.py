import logging
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


class KITTI_depth_lightning_module(pl.LightningModule):
    def __init__(self, model, loss_function, learning_rate):
        super().__init__()
        self.model = model
        self.loss_function = loss_function
        self.lr = learning_rate
        self.tstep = 0

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        if self.tstep % 10 == 0:
            imgA = transforms.ToPILImage()(x[0, :, :, :].squeeze(dim=0))
            print(type(imgA))
            wandb.log(
                {
                    "images": [
                        wandb.Image(imgA, "RGB", caption=f"Image {self.tstep} (input)"),
                        wandb.Image(
                            transforms.ToPILImage()(y[0, :, :, :].squeeze(dim=0)),
                            "RGB",
                            caption=f"Image {self.tstep} (label)",
                        ),
                        wandb.Image(
                            transforms.ToPILImage()(preds[0, :, :, :].squeeze(dim=0)),
                            "RGB",
                            caption=f"Image {self.tstep} (preds)",
                        ),
                        wandb.Image(
                            transforms.ToPILImage()((y-preds)[0, :, :, :].squeeze(dim=0)),
                            "RGB",
                            caption=f"Image {self.tstep} (y-preds)",
                        ),
                        wandb.Image(
                            transforms.ToPILImage()(torch.abs(y-preds)[0, :, :, :].squeeze(dim=0)),
                            "RGB",
                            caption=f"Image {self.tstep} abs(y-preds)",
                        ),
                    ]
                }
            )
        self.tstep += 1
        loss = self.loss_function(preds, y)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        loss = self.loss_function(preds, y)
        self.log("validation_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        loss = self.loss_function(preds, y)
        self.log("TEST loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logger = loggers.WandbLogger(project="UncertainDepths")
    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(cfg))
    model = KITTI_depth_lightning_module(
        model=BaseUNet(in_channels=3, out_channels=1),
        loss_function=nn.functional.mse_loss,
        learning_rate=cfg.hyperparameters.learning_rate,
    )
    trainer = pl.Trainer(max_epochs=1, logger=logger, limit_train_batches=0.01)
    transform = transforms.Compose([transforms.PILToTensor(),
                                    transforms.Lambda(lambda x: x/255),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),#normalize using imagenet values, as I have yet not calced it for KITTI.
                                    transforms.Pad(padding=(3, 4, 3, 5))])
    target_transform = transforms.Compose([transforms.PILToTensor(),
                                    transforms.Lambda(lambda x: x/255),
                                    transforms.Pad(padding=(3, 4, 3, 5))])
    datamodule = KITTIDataModule(
        data_dir=cfg.data_fetching_params.data_dir,
        batch_size=cfg.hyperparameters.batch_size,
        transform=transform,
        target_transform=target_transform,
        num_workers=cfg.data_fetching_params.num_workers,
    )
    
    datamodule.setup(stage="fit")
    trainer.fit(model=model, train_dataloaders=datamodule.train_dataloader(),val_dataloaders= datamodule.val_dataloader())


if __name__ == "__main__":
    main()
