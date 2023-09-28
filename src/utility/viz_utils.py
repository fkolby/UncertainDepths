import matplotlib.cm
import pdb
import wandb
import numpy as np
import torch
from torchvision import transforms


## function found from ADABINS (https://github.com/shariqfarooq123/AdaBins/blob/main/utils.pdepth)
def colorize(value, vmin=10, vmax=1000, cmap="plasma"):
    # normalize
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

    img = value[:, :, :3]
    #     return img.transpose((2, 0, 1))
    return img


def denormalize(x, device="cpu"):
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    if len(x.shape) == 3:
        return (x.unsqueeze(0) * std + mean).squeeze()
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
