import logging
import hydra
from omegaconf import OmegaConf, DictConfig
from src.models.modelImplementations.baseUNet import BaseUNet
from src.data.datamodules import KITTIDataModule
import pytorch_lightning as pl
from torch import optim, nn


class KITTI_depth(pl.LightningModule):
    def __init__(self, model, loss_function, learning_rate):
        super().__init__()
        self.model = model
        self.loss_function = loss_function
        self.lr = learning_rate

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        loss = self.loss_function(preds, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(cfg))
    loss_function = nn.functional.mse_loss
    KITTI_depth_Module = KITTI_depth(
        model=BaseUNet(in_channels=4, out_channels=2),
        loss_function=loss_function,
        learning_rate=cfg.hyperparameters.learning_rate,
    )
    trainer = pl.Trainer()
    datamodule = KITTIDataModule(data_dir=cfg.data_dir, batch_size=cfg.hyperparameters.batch_size)
    trainer.fit(model=KITTI_depth_Module, datamodule=datamodule)


if __name__ == "__main__":
    main()
