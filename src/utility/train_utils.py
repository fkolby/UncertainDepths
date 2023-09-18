import matplotlib
import hydra
from omegaconf import DictConfig, OmegaConf

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import torch
import os


def retry(how_many_tries=2):
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


def plot_and_save_tensor_as_fig(tensor: torch.Tensor, figname: str) -> None:
    print(tensor)
    if tensor.shape[0] == 1:
        plt.imshow(tensor.squeeze())
    else:
        plt.imshow(tensor.permute(1, 2, 0))
    plt.savefig(figname + ".png")
