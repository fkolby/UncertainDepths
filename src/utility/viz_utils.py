import matplotlib.cm
import pdb
import wandb
import numpy as np
import torch
from torchvision import transforms
from src.models.loss import SILogLoss
from src.utility.debug_utils import time_since_previous_log
from torch.nn import MSELoss
import time


## function found from ADABINS (https://github.com/shariqfarooq123/AdaBins/blob/main/utils.pdepth)
def colorize(value, vmin=10, vmax=1000, cmap="plasma"):
    # normalize
    value = value.cpu()
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.0
    # squeeze last dim if it exists
    # value = value.squeeze(axis=0)

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)

    img = value[:, :, :3]  # :3]
    #     return img.transpose((2, 0, 1))
    return img


def denormalize(x, device="cpu"):
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    if len(x.shape) == 3:
        return (x.unsqueeze(0).to(device=device) * std + mean).squeeze()
    elif len(x.shape) == 4:
        return x * std + mean
    else:
        err = f"Expected x to be either BxCxHxW or CxHxW, but got x with shape {x.shape}"
        raise Exception(err)


def log_images(img, depth, pred, vmin, vmax, step):
    depthdiff = colorize((depth - pred), vmin=vmin, vmax=vmax)
    absdepthdiff = colorize(torch.abs(depth - pred), vmin=vmin, vmax=vmax)
    depth = colorize(depth, vmin=vmin, vmax=vmax)

    pred = colorize(pred, vmin=vmin, vmax=vmax)
    wandb.log(
        {
            "images": [
                wandb.Image(
                    transforms.ToPILImage()(denormalize(img)),
                    "RGB",
                    caption=f"Image {step} (input)",
                ),
                wandb.Image(depth, "RGB", caption=f"Image {step} (label)"),
                wandb.Image(
                    pred,
                    "RGB",
                    caption=f"Image {step} (pred)",
                ),
                wandb.Image(
                    depthdiff,
                    "RGB",
                    caption=f"Image {step} (depth-pred)",
                ),
                wandb.Image(
                    absdepthdiff,
                    "RGB",
                    caption=f"Image {step} abs(depth-pred)",
                ),
            ]
        },
        step=step,
    )


def calc_loss_metrics(preds, targets):
    preds = preds.cpu()
    targets = targets.cpu()
    thresh = torch.maximum((targets / preds), (preds / targets))
    delta1 = (thresh < 1.25).mean(dtype=float)
    delta2 = (thresh < 1.25**2).mean(dtype=float)
    delta3 = (thresh < 1.25**3).mean(dtype=float)

    abs_rel = (torch.abs(targets - preds) / targets).mean()
    sq_rel = (((targets - preds) ** 2) / targets).mean()

    rmse = (targets - preds) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (torch.log(targets) - torch.log(preds)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = torch.log(preds) - torch.log(targets)
    silog = np.sqrt(((err**2).mean()) - err.mean() ** 2) * 100

    log_10 = (torch.abs(torch.log10(targets) - torch.log10(preds))).mean()

    silogloss_loss_func = SILogLoss()(preds, targets)
    mse_loss_func = MSELoss()(preds, targets)
    return {
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3,
        "abs_rel": abs_rel,
        "rmse": rmse,
        "log_10": log_10,
        "rmse_log": rmse_log,
        "silog": silog,
        "sq_rel": sq_rel,
        "silogloss_loss_func": silogloss_loss_func,
        "mse_loss_func": mse_loss_func,
    }


def log_loss_metrics(preds, targets, tstep=0, loss_prefix="train"):
    metrics = calc_loss_metrics(preds, targets)

    return wandb.log(
        {
            loss_prefix + "_delta1": metrics["delta1"],
            loss_prefix + "_delta2": metrics["delta2"],
            loss_prefix + "_delta3": metrics["delta3"],
            loss_prefix + "_abs_rel": metrics["abs_rel"],
            loss_prefix + "_rmse": metrics["rmse"],
            loss_prefix + "_log_10": metrics["log_10"],
            loss_prefix + "_rmse_log": metrics["rmse_log"],
            loss_prefix + "_silog": metrics["silog"],
            loss_prefix + "_sq_rel": metrics["sq_rel"],
            loss_prefix + "_SIlog_loss_func": metrics["silogloss_loss_func"],
            loss_prefix + "_MSE_loss_func": metrics["mse_loss_func"],
        },
        step=tstep,
    )
