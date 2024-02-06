import gc
import os
from pprint import pprint
import wandb

import torch
from torch.nn import MSELoss
from torchinfo import summary
from torchvision import transforms

from src.data.datamodules.KITTI_datamodule import KITTI_datamodule
from src.models.lightning_modules.base import Base_lightning_module
from src.models.modelImplementations.nnjUnet import stochastic_unet
from src.utility.train_utils import seed_everything
from src.utility.viz_utils import log_images, log_loss_metrics
from datetime import datetime

import os
from omegaconf import DictConfig, OmegaConf

from src.models.evaluate_models import eval_model
import argparse



#################### LOAD IN MODEL AND CONFIG ####################
path = "/home/jbv415/UncertainDepths/src/models/outputs/model_setup" 
parser = argparse.ArgumentParser()
parser.add_argument('ident')
args = parser.parse_args()


if args.ident.split("_")[-1]=="Laplace":
    model_type= "_".join(args.ident.split("_")[-2:])
else:
    model_type= args.ident.split("_")[-1]

cfg = OmegaConf.load(os.path.join(path,args.ident, model_type+"_config.yaml"))

print("="*40 + "\n"*2)
print(cfg)
print("\n"*2 + "="*40 )

model_path = os.path.join(cfg.save_images_path, "model_setup/", args.ident)

wandb_run_id = cfg.get("wandb_run_id")
#if wandb_run_id is not None:
    #wandb.init("ztw4gqxa", resume="must")
    #wandb.init(wandb_run_id, resume="must")
loss_function = MSELoss

model = stochastic_unet(in_channels=3, out_channels=1, cfg=cfg)
summary(model, (cfg.hyperparameters.batch_size, 3, 352, 1216), depth=300)
## free up memory again
gc.collect()
torch.cuda.empty_cache()

if cfg.models.model_type=="Ensemble":
    model.load_state_dict(torch.load(os.path.join(model_path,f"{cfg.models.model_type}_0.pt")))
else:
    model.load_state_dict(torch.load(os.path.join(model_path,f"{cfg.models.model_type}.pt")))

########################## DATALOADERS ###################################
                                
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

datamodule = KITTI_datamodule(
            transform=transform,
            target_transform=target_transform,
            cfg=cfg,
        )
datamodule.setup(stage="fit")



datamoduleEval = KITTI_datamodule(
            transform=transform,
            target_transform=target_transform,
            cfg=cfg,
        )

datamoduleEval.setup(stage="fit", dataset_type_is_ood=cfg.OOD.use_white_noise_box_test)



########################### RUN EVAL ############


if cfg.models.model_type == "Online_Laplace":
    eval_model(
                    model=model,
                    test_loader=datamoduleEval.test_dataloader(),
                    dataloader_for_hessian=datamodule.val_dataloader(),
                    cfg=cfg,
                    online_hessian=torch.load(f"{cfg.models.model_type}_hessian.pt"),
                    dont_log_wandb = True,#(wandb_run_id is None),
                ),
elif cfg.models.model_type == "Posthoc_Laplace":
    eval_model(
                    model=model,
                    test_loader=datamoduleEval.test_dataloader(),
                    dataloader_for_hessian=datamodule.val_dataloader(),
                    cfg=cfg,
                    dont_log_wandb = True,#(wandb_run_id is None),
                )
elif cfg.models.model_type == "Ensemble":
    eval_model(
                    model=model,
                    test_loader=datamoduleEval.test_dataloader(),
                    cfg=cfg,
                    dont_log_wandb = True,#(wandb_run_id is None),
                    model_path=model_path
                )
else:
    eval_model(
                    model=model,
                    test_loader=datamoduleEval.test_dataloader(),
                    cfg=cfg,
                    dont_log_wandb = True,#(wandb_run_id is None),
                )

    