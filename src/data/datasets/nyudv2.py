from src.data.datasets.base_dataset import depth_dataset
from PIL import Image
import os
from datasets import load_dataset
from torchvision.transforms import ToTensor
from omegaconf import DictConfig, OmegaConf

import pdb


class nyudv2_dataset(depth_dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.cfg = kwargs.get("cfg")
        self.dataset = load_dataset(
            "sayakpaul/nyu_depth_v2",
            trust_remote_code=True,
            num_proc=8,
            split="train",
            streaming=True,
        )
        self.train_set


""" 
    def get_PIL_image(self, *args, **kwargs):
        
        
        
        input_img = Image.open(input_path)
        label_img = Image.open(label_path)
        return input_img, label_img
 """


if __name__ == "__main__":
    cfg = OmegaConf.load("/home/jbv415/UncertainDepths/src/conf/config.yaml")

    cfg["dataset_params"] = {
        "name": "nyudv2",
        "test_set_is_actually_valset": True,
        "num_workers": 8,
        "min_depth ": 1e-8,
        "max_depth ": 10,
        "input_height": 352,
        "input_width": 1216,
    }

    nyu = nyudv2_dataset(
        train_or_test="train",
        transform=ToTensor(),
        target_transform=ToTensor(),
        cfg=cfg,
    )
    print("done")

    print(nyu)
    print(dir(nyu))
    pdb.set_trace()
