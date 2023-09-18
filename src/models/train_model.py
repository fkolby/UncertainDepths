import logging
import hydra
from omegaconf import OmegaConf, DictConfig
from src.models.modelImplementations.baseUNet import BaseUNet
from src.data.datamodules import KITTIDataModule
import pytorch_lightning as pl
from torch import optim, nn
import pdb
class KITTI_depth_lightning_module(pl.LightningModule):
    def __init__(self, model, loss_function, learning_rate):
        super().__init__()
        self.model = model
        self.loss_function = loss_function
        self.lr = learning_rate

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        print(f"shape of x: {x.shape}, yshape: {y.shape}, shape of preds: {preds.shape}")
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
    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(cfg))
    loss_function = nn.functional.mse_loss
    model = KITTI_depth_lightning_module(
        model=BaseUNet(in_channels=3, out_channels=1),
        loss_function=loss_function,
        learning_rate=cfg.hyperparameters.learning_rate,
    )
    trainer = pl.Trainer(num_sanity_val_steps=0)

    datamodule = KITTIDataModule(data_dir=cfg.data_dir, batch_size=cfg.hyperparameters.batch_size)
    debugdatamodule = KITTIDataModule(data_dir=cfg.data_dir, batch_size=cfg.hyperparameters.batch_size)
    debugdatamodule.setup("fit")
    print(type(debugdatamodule.train_dataloader()))
    print(iter(debugdatamodule.train_dataloader()).__next__())
    print("is dataloader iterable?", hasattr(debugdatamodule.train_dataloader(),"__iter__"))
    print("now comes error")
    datamodule.setup(stage="fit")
    trainer.fit(model=model, train_dataloaders= datamodule.train_dataloader())


if __name__ == "__main__":
    main()
