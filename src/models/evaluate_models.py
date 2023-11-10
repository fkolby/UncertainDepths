import torch
from tqdm import tqdm
import pdb
from src.utility.debug_utils import time_since_previous_log
from src.models.predict_model import infer
from src.utility.other_utils import RunningAverageDict
from src.utility.eval_utils import compute_metrics
import numpy as np
import os
import timeit
import wandb
from PIL import Image
import torchvision.transforms as transforms
from src.utility.viz_utils import colorize, denormalize


@torch.no_grad()
def eval_model(model, model_name, test_loader, config, round_vals=True, round_precision=3):
    metrics = RunningAverageDict()
    i=0
    for sample in tqdm(test_loader, total=len(test_loader)):
        if "has_valid_depth" in sample:
            if not sample["has_valid_depth"]:
                print("validDEPTH?")
                continue
        images, _, depths = sample
        images = images.to(device="cuda")
        depths = depths.to(device="cuda")

        if config.model_type == "Ensemble": # Model is then actually a dict of models:
            j = 0
            preds = torch.zeros(size=(config.n_models,depths.shape[0],depths.shape[1],config.dataset_params.input_height, config.dataset_params.input_width)) #dimensions: model, batch, color, height, width.
            for name, mod in model.items():
                
                preds[j,:,:,:,:] = torch.unsqueeze(infer(mod, images), dim = 0) #unsqueeze to have a model-dimension (first)
                j+=1
        if config.model_type == "Dropout": #Only one model, but is different every call. 
            preds = torch.zeros(size=(config.n_models,depths.shape[0],depths.shape[1],config.dataset_params.input_height, config.dataset_params.input_width)) #dimensions: model, batch, color, height, width.
            for j in range(config.n_models): 
                preds[j,:,:,:,:] = torch.unsqueeze(infer(model, images), dim = 0) #unsqueeze to have a model-dimension (first)
        
        if config.model_type == "BaseUNet":
            preds = torch.zeros(size=(config.n_models,depths.shape[0],depths.shape[1],config.dataset_params.input_height, config.dataset_params.input_width)) #dimensions: model, batch, color, height, width.
            for j in range(config.n_models):
                preds[j,:,:,:,:] = torch.unsqueeze(infer(model, images), dim = 0) #unsqueeze to have a model-dimension (first)
        


        # Save image, depth, pred for visualization
        if config.save_images and i == 0:
            for j in range(min(images.shape[0], 6)):
                image = images[j, :, :, :]
                depth = depths[j, :, :, :]
                pred = torch.mean(preds[:,j, :, :, :], dim=0) #pred is average prediction
                var = torch.mean(preds[:,j, :, :, :]**2, dim=0) - pred**2 #var is variance over model. since color dimension is 1, it is by pixel.

                os.makedirs(config.save_images_path, exist_ok=True)
                d = torch.tensor(np.transpose(colorize(torch.squeeze(depth,dim=0), 0, 80), (2,0,1)))
                p = torch.tensor(np.transpose(colorize(torch.squeeze(pred,dim=0), 0, 80), (2,0,1)))
                v = torch.tensor(np.transpose(colorize(torch.squeeze(var,dim=0), 0, 80), (2,0,1)))
                im = transforms.ToPILImage()(denormalize(image).cpu())
                print(type(d))

                im.save(os.path.join(config.save_images_path, f"{j}_{model_name}_img.png"))
                print(d.shape, p.shape, image.shape, pred.shape, depth.shape)
                transforms.ToPILImage()(d).save(
                    os.path.join(config.save_images_path, f"{j}_{model_name}_depth.png")
                )
                transforms.ToPILImage()(p).save(
                    os.path.join(config.save_images_path, f"{j}_{model_name}_pred.png")
                )
                transforms.ToPILImage()(v).save(
                    os.path.join(config.save_images_path, f"{j}_{model_name}_var.png")
                )


                np.save(
                    os.path.join(config.save_images_path, f"np_img_{model_name}_{j}.npy"),
                    torch.squeeze(denormalize(image), dim=0).numpy(force=True),
                )
                np.save(
                    os.path.join(config.save_images_path, f"np_depth_{model_name}_{j}.npy"),
                    torch.squeeze(depth, dim=0).numpy(force=True),
                )
                np.save(
                    os.path.join(config.save_images_path, f"np_preds_{model_name}_{j}.npy"),
                    torch.squeeze(pred, dim=0).numpy(force=True),
                )
        if config.in_debug and i > 0:
            continue
        i += 1

        preds = torch.mean(preds, dim=0) #pred is average prediction
        # print(depth.shape, pred.shape)

        losses = compute_metrics(depths, preds, config=config)
        metrics.update(losses)
    if round_vals:

        def r(m):
            return torch.round(m, decimals=round_precision)

    else:

        def r(m):
            return m

    metrics = {k: r(v) for k, v in metrics.get_value().items()}
    return metrics
