import os
import pdb
import random

import hydra
import matplotlib
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig, OmegaConf


def retry(how_many_tries: int = 2):
    """A decorator, which retries to run a function how_many_tries times"""

    def wrapper(func):
        def try_it(*args, **kwargs):
            tries = 0
            while tries < how_many_tries:
                try:
                    return func(*args, **kwargs)
                except:
                    tries += 1
            return -1

        return try_it

    return wrapper


def seed_everything(seed: int):
    # taken directly from zoedepth
    import random

    import numpy
    import torch

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_and_save_tensor_as_fig(tensor: torch.Tensor, figname: str) -> None:
    print(tensor)
    if tensor.shape[0] == 1:
        plt.imshow(tensor.squeeze())
    else:
        plt.imshow(tensor.permute(1, 2, 0))
    plt.savefig(figname + ".png")


# Shamelessly stolen from adabins repo( https://github.com/shariqfarooq123/AdaBins/blob/main/dataloader.py) but added doc.
def random_crop(img: torch.Tensor, depth: torch.Tensor, height: int, width: int):
    """Crops a image randomly."""
    try:
        assert img.shape[1] >= height
        assert img.shape[2] >= width
        assert img.shape[1] == depth.shape[1]
        assert img.shape[2] == depth.shape[2]
    except:
        pdb.set_trace()
    x = random.randint(0, img.shape[2] - width)
    y = random.randint(0, img.shape[1] - height)
    img = img[:, y : y + height, x : x + width]
    depth = depth[:, y : y + height, x : x + width]
    return img, depth
