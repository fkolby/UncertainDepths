import pdb
import time

import matplotlib.cm
import numpy as np
import torch
from torch.nn import MSELoss
from torchvision import transforms

import wandb
from src.models.loss import SILogLoss
from src.utility.debug_utils import time_since_previous_log


## function found from ADABINS (https://github.com/shariqfarooq123/AdaBins/blob/main/utils.pdepth)
def colorize(
    value: torch.Tensor, vmin: float = 10.0, vmax: float = 1000.0, cmap="plasma"
) -> torch.Tensor:
    """Colorize an image"""
    # normalize
    print("colorize")
    print(value.shape)
    value = value.squeeze()
    if len(value.shape) != 2:
        print(
            "Wrong shape in colorize - shapes should (after squeezing) be 2D. Your shape after squeezing is:",
            value.shape,
            flush=True,
        )
        raise NotImplementedError
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
    print(img.shape)
    return img


def denormalize(x: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """Denormalizes a batch of images, based on imagenet statistics."""

    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    if len(x.shape) == 3:
        out = (x.unsqueeze(0).to(device=device) * std + mean).squeeze()
    elif len(x.shape) == 4:
        out = x * std + mean
    else:
        err = f"Expected x to be either BxCxHxW or CxHxW, but got x with shape {x.shape}"
        raise Exception(err)
    return torch.clamp(out, 0, 1)


def log_images(
    img: torch.Tensor,
    depth: torch.Tensor,
    pred: torch.Tensor,
    vmin: float,
    vmax: float,
    step: int,
    image_appendix="",
) -> None:
    """Colorizes grayscale depth and pred and logs differences and abs differences of depth and predicition. Logs result to wandb."""

    depthdiff = colorize((depth - pred), vmin=vmin, vmax=vmax)
    absdepthdiff = colorize(torch.abs(depth - pred), vmin=vmin, vmax=vmax)
    depth = colorize(depth, vmin=vmin, vmax=vmax)

    img = img.squeeze()

    pred = colorize(pred, vmin=vmin, vmax=vmax)
    if step > 5e9:
        print(
            denormalize(img.shape), depth.shape, pred.shape
        )  # debugging error message of non-matching image sizes - error in WandB UI
    wandb.log(
        {
            "images"
            + image_appendix: [
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
    """Calculates relevant loss metrics - actual metrics"""
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


def log_loss_metrics(
    preds: torch.Tensor, targets: torch.Tensor, tstep: int = 0, loss_prefix="train"
) -> None:
    """Calcs loss metrics and logs them to wandb"""
    metrics = calc_loss_metrics(preds=preds, targets=targets)
    wandb.log(
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
