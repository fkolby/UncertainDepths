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

from src.models.lightning_modules.base import Base_module


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
    ):  # if ensemble we need to seed - and therefore instantiate dataloaders seperately (they should have different seed for every model)( Zoe does not use a training set)
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

    if cfg.models.model_type == "ZoeNK":
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

    elif cfg.models.model_type == "Ensemble":
        seed_everything(cfg.seed)
        # Zoe does not want normalization (does it internally), Ensemble needs new seed for each run
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
        # Zoe does not want normalization (does it internally), Ensemble needs new seed for each run
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

            model = Base_module(
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
                    test_loader=datamoduleEval.val_dataloader(),
                    dataloader_for_hessian=datamodule.train_dataloader(),
                    cfg=cfg,
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

                neuralnet = stochastic_unet(in_channels=3, out_channels=1, cfg=cfg)

                summary(neuralnet, (1, 3, 352, 704), depth=300)

                model = Base_module(
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

            seed_everything(
                seed=cfg.seed
            )  # seed to same, so first pictures match (and thus example pictures for showing variance are similar)
            model = stochastic_unet(
                in_channels=3, out_channels=1, cfg=cfg
            )  # just loading it in, to have something to passe to eval_model function. this specific model gets overwritten in eval_model.

            #### EDIT IN EVAL_MODEL FOR ENSEMBLES.
            model.load_state_dict(torch.load(f"{cfg.models.model_type}_0.pt"))

            model.eval().to("cuda")
            pprint(
                eval_model(
                    model=model,
                    test_loader=datamoduleEval.val_dataloader(),
                    cfg=cfg,
                )
            )

        case "Dropout":  # main difference to laplace posthoc is the fact that we do not put module into eval mode.
            neuralnet = stochastic_unet(in_channels=3, out_channels=1, cfg=cfg)
            summary(neuralnet, (1, 3, 352, 704), depth=300)

            model = Base_module(
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

            model.to("cuda")
            pprint(
                eval_model(
                    model=model,
                    test_loader=datamoduleEval.val_dataloader(),
                    cfg=cfg,
                )
            )

        case "BaseUNet":
            raise NotImplementedError
            model = Base_module(
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
                    cfg=cfg,
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
                    cfg=cfg,
                )
            )


if __name__ == "__main__":
    main()
