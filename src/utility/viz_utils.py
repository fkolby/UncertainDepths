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


def log_images(img, depth, pred, vmin, vmax, step):
    depthdiff = colorize((depth - pred), vmin=vmin, vmax=vmax)
    absdepthdiff = colorize(torch.abs(depth - pred), vmin=vmin, vmax=vmax)
    depth = colorize(depth, vmin=vmin, vmax=vmax)
    pred = colorize(pred, vmin=vmin, vmax=vmax)
    wandb.log(
        {
            "images": [
                wandb.Image(transforms.ToPILImage()(img), "RGB", caption=f"Image {step} (input)"),
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
