import argparse
import gc
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
from src.data.datamodules.KITTI_datamodule import KITTI_datamodule
from src.models.evaluate_models import eval_model
from src.models.lightning_modules.base import Base_lightning_module
from src.models.loss import SILogLoss
from src.models.modelImplementations.baseUNet import BaseUNet
from src.models.modelImplementations.nnjUnet import stochastic_unet
from src.utility.train_utils import seed_everything
from src.utility.viz_utils import log_images, log_loss_metrics
from datetime import datetime


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Trains a model, given by the config. Supported setups are currently KITTI as dataset, and either MCDropout, Ensemble, Online Laplace or Posthoc Laplace"""

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
        trainer_args = {
            "max_epochs": cfg.trainer_args.max_epochs,
            "gradient_clip_val": 1.0,
        }

    slurm_id = str(os.environ.get("SLURM_JOB_ID"))

    wandb_run = wandb.init(
        project="UncertainDepths",
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        name=cfg.models.model_type
        + "_"
        + "slurm_"
        + slurm_id
        + "_"
        + datetime.now().strftime("%d_%m_%Y_%H"),
    )
    OmegaConf.update(cfg, "wandb_run_id", wandb_run._run_id, force_add=True)

    wandb_run.log_code(
        "~/UncertainDepths/src",
        include_fn=lambda path: path.endswith(".py")
        and not (
            path.contains("/pytorch-laplace/")
            or path.contains("/nnj/")
            or path.contains("/wandb/")
            or path.contains("cache/")
        ),
    )
    logger = loggers.WandbLogger(project="UncertainDepths")
    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(cfg))

    date_and_time_and_model = (
        datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + "_" + slurm_id + "_" + cfg.models.model_type
    )

    model_path = os.path.join(cfg.save_images_path, "model_setup/", date_and_time_and_model)

    os.makedirs(model_path)

    # =========================== TRANSFORMS & DATAMODULES ===============================================
    seed_everything(cfg.seed)
    transform = transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.Lambda(lambda x: x / 255),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # normalize using imagenet values, as I have yet not calced it for KITTI.
        ]
    )
    target_transform = transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.Lambda(lambda x: x / 256),  # 256 as per devkit
        ]
    )

    if (
        cfg.models.model_type != "Ensemble"
    ):  # if ensemble we need to seed - and therefore instantiate dataloaders seperately (they should have different seed for every model)( Zoe does not use a training set)
        datamodule = KITTI_datamodule(
            transform=transform,
            target_transform=target_transform,
            cfg=cfg,
        )
        datamodule.setup(stage="fit")

    if cfg.models.model_type == "ZoeNK":
        seed_everything(cfg.seed)
        # no normalization, just straight up load in. (apart from center-crop and randomcrop)
        datamoduleEval = KITTI_datamodule(
            transform=transforms.Compose(
                [
                    transforms.PILToTensor(),
                ]
            ),
            target_transform=target_transform,
            cfg=cfg,
        )
        datamoduleEval.setup(stage="fit", dataset_type_is_ood=cfg.OOD.use_white_noise_box_test)
    else:
        seed_everything(cfg.seed)
        datamoduleEval = KITTI_datamodule(
            transform=transform,
            target_transform=target_transform,
            cfg=cfg,
        )

        datamoduleEval.setup(stage="fit", dataset_type_is_ood=cfg.OOD.use_white_noise_box_test)

    # ================================ SET LOSS FUNC ======================================================

    if cfg.loss_function == "MSELoss":
        loss_function = MSELoss()
    else:
        raise NotImplementedError

    # ================================ TRAIN & EVAL ========================================================
    match cfg.models.model_type:
        case "Posthoc_Laplace":
            neuralnet = stochastic_unet(in_channels=3, out_channels=1, cfg=cfg)
            summary(neuralnet, (1, 3, 352, 1216), depth=300)

            model = Base_lightning_module(
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
            trainer.save_checkpoint(os.path.join(model_path, f"{cfg.models.model_type}.ckpt"))
            # now we dont need (or want) lightning anymore
            torch.save(
                model._modules["model"].state_dict(),
                os.path.join(model_path, f"{cfg.models.model_type}.pt"),
            )
            OmegaConf.save(
                config=cfg, f=os.path.join(model_path, f"{cfg.models.model_type}_config.yaml")
            )

            ## free up memory again
            del trainer
            del model
            del neuralnet
            gc.collect()
            torch.cuda.empty_cache()

            model = stochastic_unet(in_channels=3, out_channels=1, cfg=cfg)
            model.load_state_dict(torch.load(f"{cfg.models.model_type}.pt"))

            model.eval().to("cuda")
            wandb.log(
                eval_model(
                    model=model,
                    test_loader=datamoduleEval.test_dataloader(),
                    dataloader_for_hessian=datamodule.val_dataloader(),
                    cfg=cfg,
                ),
                step=50000,
            )
        case "Online_Laplace":
            neuralnet = stochastic_unet(in_channels=3, out_channels=1, cfg=cfg)
            summary(neuralnet, (1, 3, 352, 1216), depth=300)

            model = Base_lightning_module(
                model=neuralnet,
                loss_function=loss_function,
                cfg=cfg,
                steps_per_epoch=len(datamodule.train_dataloader()),
                dataset_size=datamodule.train_dataloader().dataset.__len__(),
            )

            trainer = pl.Trainer(logger=logger, **trainer_args)

            trainer.fit(
                model=model,
                train_dataloaders=datamodule.train_dataloader(),
                val_dataloaders=datamodule.val_dataloader(),
            )
            trainer.save_checkpoint(os.path.join(model_path, f"{cfg.models.model_type}.ckpt"))
            # now we dont need (or want) lightning anymore
            torch.save(
                model._modules["model"].state_dict(),
                os.path.join(model_path, f"{cfg.models.model_type}.pt"),
            )
            OmegaConf.save(
                config=cfg, f=os.path.join(model_path, f"{cfg.models.model_type}_config.yaml")
            )

            print(type(trainer.model.Online_Laplace))
            trainer.model.Online_Laplace.save_hessian(
                os.path.join(model_path, f"{cfg.models.model_type}_hessian.pt")
            )
            ## free up memory again
            del trainer
            del model
            del neuralnet
            gc.collect()
            torch.cuda.empty_cache()

            model = stochastic_unet(in_channels=3, out_channels=1, cfg=cfg)
            model.load_state_dict(torch.load(f"{cfg.models.model_type}.pt"))

            model.eval().to("cuda")
            wandb.log(
                eval_model(
                    model=model,
                    test_loader=datamoduleEval.test_dataloader(),
                    dataloader_for_hessian=datamodule.val_dataloader(),
                    cfg=cfg,
                    online_hessian=torch.load(f"{cfg.models.model_type}_hessian.pt"),
                ),
                step=500000,
            )

        case "Ensemble":
            seed_everything(cfg.seed)
            seeds = [np.random.randint(0, 10000) for i in range(cfg.models.n_models)]
            for i in range(cfg.models.n_models):
                seed_everything(seeds[i])  # Seed both dataloaders and neural net initialization.

                datamodule = KITTI_datamodule(
                    transform=transform,
                    target_transform=target_transform,
                    cfg=cfg,
                )
                datamodule.setup(stage="fit")

                neuralnet = stochastic_unet(in_channels=3, out_channels=1, cfg=cfg)

                summary(neuralnet, (1, 3, 352, 1216), depth=300)

                model = Base_lightning_module(
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

                trainer.save_checkpoint(os.path.join(model_path, f"{cfg.models.model_type}.ckpt"))
                # now we dont need (or want) lightning anymore
                torch.save(
                    model._modules["model"].state_dict(),
                    os.path.join(model_path, f"{cfg.models.model_type}_{i}.pt"),
                )
                OmegaConf.save(
                    config=cfg, f=os.path.join(model_path, f"{cfg.models.model_type}_config.yaml")
                )

                ## free up memory again
                del trainer
                del model
                del neuralnet
                gc.collect()
                torch.cuda.empty_cache()

                print(torch.cuda.memory_summary())

            seed_everything(
                seed=cfg.seed
            )  # seed to same, so first pictures match (and thus example pictures for showing variance are similar)
            model = stochastic_unet(
                in_channels=3, out_channels=1, cfg=cfg
            )  # just loading it in, to have something to passe to eval_model function. this specific model gets overwritten in eval_model.

            #### EDIT IN EVAL_MODEL FOR ENSEMBLES.
            model.load_state_dict(torch.load(f"{cfg.models.model_type}_0.pt"))

            model.eval().to("cuda")
            wandb.log(
                eval_model(
                    model=model,
                    test_loader=datamoduleEval.test_dataloader(),
                    cfg=cfg,
                ),
                step=50000,
            )

        case "Dropout":  # main difference to laplace posthoc is the fact that we do not put module into eval mode.
            neuralnet = stochastic_unet(in_channels=3, out_channels=1, cfg=cfg)
            summary(neuralnet, (1, 3, 352, 1216), depth=300)

            model = Base_lightning_module(
                model=neuralnet,
                loss_function=loss_function,
                cfg=cfg,
                steps_per_epoch=len(datamodule.train_dataloader()),
            )

            trainer = pl.Trainer(logger=logger, **trainer_args)

            trainer.fit(
                model=model,
                train_dataloaders=datamodule.train_dataloader(),
                val_dataloaders=datamodule.test_dataloader(),
            )

            trainer.save_checkpoint(os.path.join(model_path, f"{cfg.models.model_type}.ckpt"))
            # now we dont need (or want) lightning anymore
            torch.save(
                model._modules["model"].state_dict(),
                os.path.join(model_path, f"{cfg.models.model_type}.pt"),
            )
            OmegaConf.save(
                config=cfg, f=os.path.join(model_path, f"{cfg.models.model_type}_config.yaml")
            )

            # free up memory
            del trainer
            del model
            del neuralnet
            gc.collect()
            torch.cuda.empty_cache()

            # dont want/need lightning now
            model = stochastic_unet(in_channels=3, out_channels=1, cfg=cfg)
            model.load_state_dict(torch.load(f"{cfg.models.model_type}.pt"))

            model.to("cuda")
            wandb.log(
                eval_model(
                    model=model,
                    test_loader=datamoduleEval.test_dataloader(),
                    cfg=cfg,
                ),
                step=50000,
            )

        case "BaseUNet":
            raise NotImplementedError
            model = Base_lightning_module(
                model=BaseUNet(in_channels=3, out_channels=1, cfg=cfg),
                loss_function=loss_function,
                cfg=cfg,
                steps_per_epoch=len(datamodule.train_dataloader()),
            )

            trainer = pl.Trainer(logger=logger, **trainer_args)

            trainer.fit(
                model=model,
                train_dataloaders=datamodule.train_dataloader(),
                val_dataloaders=datamodule.test_dataloader(),
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
            wandb.log(
                eval_model(
                    model=model,
                    model_name="Unet",
                    test_loader=datamoduleEval.test_dataloader(),
                    cfg=cfg,
                ),
                step=50000,
            )
        case "ZoeNK":
            repo = "isl-org/ZoeDepth"
            ZoeNK = torch.hub.load(repo, "ZoeD_NK", pretrained=True).eval().to("cuda")
            print("now it errors")
            wandb.log(
                eval_model(
                    model=ZoeNK,
                    model_name="ZoeNK",
                    test_loader=datamoduleEval.test_dataloader(),
                    cfg=cfg,
                ),
                step=500000,
            )


if __name__ == "__main__":
    main()
