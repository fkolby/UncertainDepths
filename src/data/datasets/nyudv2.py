from src.data.datasets.base_dataset import depth_dataset
from PIL import Image
import os
from datasets import load_dataset
from torchvision.transforms import ToTensor
from omegaconf import DictConfig, OmegaConf


class nyudv2_dataset(depth_dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.cfg = kwargs.get("cfg")
        self.train_dataset = load_dataset("sayakpaul/nyu_depth_v2", trust_remote_code=True)

        

""" 
    def get_PIL_image(self, *args, **kwargs):
        
        
        
        input_img = Image.open(input_path)
        label_img = Image.open(label_path)
        return input_img, label_img
 """
cfg = OmegaConf.load("/home/jbv415/UncertainDepths/src/conf/config.yaml")
print(cfg)


if __name__=="__main__":
    nyudv2_dataset(train_or_test="train",
                transform=ToTensor(),
                target_transform=ToTensor(),
                cfg=cfg,)
    print("done")