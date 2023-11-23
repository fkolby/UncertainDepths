import argparse
import logging
import os
import pdb
from pprint import pprint

import hydra
import numpy as np
import PIL
import pytorch_lightning as pl
import seaborn as sns
import torch
from lightning.pytorch import callbacks, loggers
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import callbacks
from torch import nn, optim
from torch.nn import MSELoss
from torchinfo import summary
from torchvision import transforms

import wandb
from src.data.datamodules import KITTIDataModule
from src.models.evaluate_models import eval_model
from src.models.loss import SILogLoss
from src.models.modelImplementations.baseUNet import BaseUNet
from src.models.modelImplementations.nnjUnet import stochastic_unet
from src.utility.train_utils import seed_everything
from src.utility.viz_utils import log_images, log_loss_metrics


class KITTI_depth_lightning_module(pl.LightningModule):
    def __init__(
        self,
        model,
        loss_function,
        steps_per_epoch: int,
        cfg,
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
        self.steps_per_epoch = steps_per_epoch

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        x, y, fullsize_targets = batch

        try:
            assert (x[:, 0, :, :].shape == y[:, 0, :, :].shape) & (
                x[0, 0, :, :].shape == torch.Size((self.input_height, self.input_width))
            )

        except:
            pdb.set_trace()
        preds = self(x)
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
        wandb.log(
            {"train_loss": loss, "learning_rate": self.lr_schedulers().get_last_lr()[0]},
            step=self.tstep,
        )

        fullsize_mask = torch.logical_and(
            fullsize_targets > self.min_depth, fullsize_targets < self.max_depth
        )

        masked_full_size_targets = fullsize_targets[fullsize_mask]

        resized_preds = nn.functional.interpolate(
            preds, fullsize_targets.shape[-2:], mode="bilinear", align_corners=True
        )
        masked_resized_preds = resized_preds[fullsize_mask]
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
        preds = self(x)
        print(f"VALIDATION: x: {x.shape} y: {y.shape}, pred: {preds.shape}")
        mask = torch.logical_and(y > self.min_depth, y < self.max_depth)
        loss = self.loss_function(preds * mask, y * mask)

        wandb.log({"val_loss": loss}, step=self.tstep)
        self.log("validation_loss", loss)

        fullsize_mask = torch.logical_and(
            fullsize_targets > self.min_depth, fullsize_targets < self.max_depth
        )

        masked_full_size_targets = fullsize_targets[fullsize_mask]

        resized_preds = nn.functional.interpolate(
            preds, fullsize_targets.shape[-2:], mode="bilinear", align_corners=True
        )

        masked_resized_preds = resized_preds[fullsize_mask]
        log_loss_metrics(
            preds=masked_resized_preds.detach(),
            targets=masked_full_size_targets.detach(),
            tstep=self.tstep,
            loss_prefix="val",
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        mask = torch.logical_and(y > self.min_depth, y < self.max_depth)
        loss = self.loss_function(preds * mask, y * mask)
        self.log("TEST loss", loss)
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


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print("GPU: ", torch.cuda.is_available())
    print("nGPUs:", torch.cuda.device_count())
    print("CurrDevice:", torch.cuda.current_device())
    print("Device name: ", torch.cuda.get_device_name(0))
    print(torch.cuda.memory_summary())
    # ================================SETUP ========================================================
    if cfg.in_debug:
        if not cfg.pdb_disabled:
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

    # =========================== TRANSFORMS & DATAMODULES ===============================================

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
    if (
        cfg.models.model_type != "Ensemble"
    ):  # if ensemble we need to seed - and therefore instantiate dataloaders seperately (they should have different seed for every model)( Zoe does not use this)
        seed_everything(cfg.seed)

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

    if cfg.models.model_type not in [
        "ZoeNK",
        "Ensemble",
    ]:  # Zoe does not want normalization (does it internally), Ensemble needs new seed for each run
        datamoduleEval = KITTIDataModule(
            data_dir=cfg.dataset_params.data_dir,
            batch_size=cfg.hyperparameters.batch_size,
            transform=transform,
            target_transform=target_transform,
            num_workers=cfg.dataset_params.num_workers,
            input_height=cfg.dataset_params.input_height,
            input_width=cfg.dataset_params.input_width,
            pytorch_lightning_in_use=False,  # KEY ARGUMENT HERE FOR SPEED.
        )

        datamoduleEval.setup(stage="fit")
    else:
        # no normalization, just straight up load in. (apart from center-crop and randomcrop)
        datamoduleEval = KITTIDataModule(
            data_dir=cfg.dataset_params.data_dir,
            batch_size=cfg.hyperparameters.batch_size,
            transform=transforms.Compose(
                [transforms.PILToTensor(), transforms.CenterCrop((352, 1216))]
            ),
            target_transform=target_transform,
            num_workers=cfg.dataset_params.num_workers,
            input_height=cfg.dataset_params.input_height,
            input_width=cfg.dataset_params.input_width,
            pytorch_lightning_in_use=False,  # KEY ARGUMENT HERE FOR SPEED.
        )
        datamoduleEval.setup(stage="fit")

    # ================================ SET LOSS FUNC ======================================================

    if cfg.loss_function == "MSELoss":
        loss_function = MSELoss()
    else:
        raise NotImplementedError

    # ================================ TRAIN & EVAL ========================================================
    match cfg.models.model_type:
        case "stochastic_unet":
            neuralnet = stochastic_unet(in_channels=3, out_channels=1, cfg=cfg)
            summary(neuralnet, (1, 3, 352, 704), depth=300)

            model = KITTI_depth_lightning_module(
                model=neuralnet,
                loss_function=loss_function,
                cfg=cfg,
                steps_per_epoch=len(datamodule.train_dataloader()),
            )

            trainer = pl.Trainer(logger=logger, **trainer_args)

            trainer.fit(
                model=model,
                train_dataloaders=datamodule.train_dataloader(),
                val_dataloaders=datamodule.val_dataloader(),
            )
            trainer.save_checkpoint(f"{cfg.models.model_type}.ckpt")
            # now we dont need (or want) lightning anymore
            torch.save(
                model._modules["model"].state_dict(),
                f"{cfg.models.model_type}.pt",
            )
            model = stochastic_unet(in_channels=3, out_channels=1, cfg=cfg)
            model.load_state_dict(torch.load(f"{cfg.models.model_type}.pt"))

            model.eval().to("cuda")
            pprint(
                eval_model(
                    model=model,
                    model_name="stochastic_unet",
                    test_loader=datamoduleEval.val_dataloader(),
                    dataloader_for_hessian=datamodule.val_dataloader(),  # when in actual (final) test, we should do val_loader here, but test_loader for test_loader.
                    test_data_img_nums=len(datamoduleEval.KITTI_val_set),
                    config=cfg,
                )
            )
        case "Ensemble":
            for i in range(cfg.models.n_models):
                seed_everything(
                    cfg.seed + i
                )  # Seed both dataloaders and neural net initialization.

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

                datamoduleEval = KITTIDataModule(
                    data_dir=cfg.dataset_params.data_dir,
                    batch_size=cfg.hyperparameters.batch_size,
                    transform=transform,
                    target_transform=target_transform,
                    num_workers=cfg.dataset_params.num_workers,
                    input_height=cfg.dataset_params.input_height,
                    input_width=cfg.dataset_params.input_width,
                    pytorch_lightning_in_use=False,  # KEY ARGUMENT HERE FOR SPEED.
                )

                datamoduleEval.setup(stage="fit")

                neuralnet = stochastic_unet(in_channels=3, out_channels=1, cfg=cfg)
                summary(neuralnet, (1, 3, 352, 704), depth=300)

                model = KITTI_depth_lightning_module(
                    model=neuralnet,
                    loss_function=loss_function,
                    cfg=cfg,
                    steps_per_epoch=len(datamodule.train_dataloader()),
                )

                trainer = pl.Trainer(logger=logger, **trainer_args)

                trainer.fit(
                    model=model,
                    train_dataloaders=datamodule.train_dataloader(),
                    val_dataloaders=datamodule.val_dataloader(),
                )
                trainer.save_checkpoint(f"{cfg.models.model_type}.ckpt")
                # now we dont need (or want) lightning anymore
                torch.save(
                    model._modules["model"].state_dict(),
                    f"{cfg.models.model_type}_{i}.pt",
                )
            model = stochastic_unet(
                in_channels=3, out_channels=1, cfg=cfg
            )  # just loading it in, to have something to passe to eval_model function. this specific model gets overwritten in eval_model.

            #### EDIT IN EVAL_MODEL FOR ENSEMBLES.
            model.load_state_dict(torch.load(f"{cfg.models.model_type}.pt"))

            model.eval().to("cuda")
            pprint(
                eval_model(
                    model=model,
                    model_name="stochastic_unet",
                    test_loader=datamoduleEval.val_dataloader(),
                    dataloader_for_hessian=datamodule.val_dataloader(),  # when in actual (final) test, we should do val_loader here, but test_loader for test_loader.
                    test_data_img_nums=len(datamoduleEval.KITTI_val_set),
                    config=cfg,
                )
            )
        case "BaseUNet":
            raise NotImplementedError
            model = KITTI_depth_lightning_module(
                model=BaseUNet(in_channels=3, out_channels=1, cfg=cfg),
                loss_function=loss_function,
                cfg=cfg,
                steps_per_epoch=len(datamodule.train_dataloader()),
            )

            trainer = pl.Trainer(logger=logger, **trainer_args)

            trainer.fit(
                model=model,
                train_dataloaders=datamodule.train_dataloader(),
                val_dataloaders=datamodule.val_dataloader(),
            )
            trainer.save_checkpoint(f"{cfg.models.model_type}.ckpt")
            # now we dont need (or want) lightning anymore
            torch.save(
                model._modules["model"].state_dict(),
                f"{cfg.models.model_type}.pt",
            )
            model = stochastic_unet(in_channels=3, out_channels=1, cfg=cfg)
            model.load_state_dict(torch.load(f"{cfg.models.model_type}.pt"))

            model.eval().to("cuda")
            pprint(
                eval_model(
                    model=model,
                    model_name="Unet",
                    test_loader=datamoduleEval.val_dataloader(),
                    test_data_img_nums=len(datamoduleEval.KITTI_val_set),
                    config=cfg,
                )
            )
        case "ZoeNK":
            repo = "isl-org/ZoeDepth"
            ZoeNK = torch.hub.load(repo, "ZoeD_NK", pretrained=True).eval().to("cuda")
            print("now it errors")
            pprint(
                eval_model(
                    model=ZoeNK,
                    model_name="ZoeNK",
                    test_loader=datamoduleEval.val_dataloader(),
                    test_data_img_nums=len(datamoduleEval.KITTI_val_set),
                    config=cfg,
                )
            )


if __name__ == "__main__":
    main()
