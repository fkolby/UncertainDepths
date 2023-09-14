import lightning.pytorch as pl
import numpy as np
from src.utility.train_utils import plot_and_save_tensor_as_fig
from PIL import Image
import pandas as pd
import os
from torchvision import transforms
import subprocess
from torch.utils.data import DataLoader, Dataset,Subset
import csv

def extract_val_paths(homepath: str, train_or_val: str, out_file: str) -> None:
    # train_or_val_path is e.g. "~..../external/KITTI/"
    """This function writes all the paths of images in train_or_val_path following the download to out_file"""
    train_or_val_path = homepath + train_or_val + "/"
    if os.path.isdir(out_file):
        raise NameError("You a directory as file to write to.")
    if os.path.exists(out_file):
        os.remove(out_file)
        os.system("touch " + out_file)
    else:
        os.system("touch " + out_file)
    with open(out_file, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["input_path", "label_path"])
        for datesync in (
            subprocess.check_output("ls " + train_or_val_path, shell=True)
            .decode("utf-8")
            .split("\n")
        ):
            if len(datesync) == 0:
                continue
            for imageLeftRight in (
                subprocess.check_output(
                    "ls " + train_or_val_path + "/" + datesync + "/" + "proj_depth/groundtruth/",
                    shell=True,
                )
                .decode("utf-8")
                .split("\n")
            ):
                if len(imageLeftRight) == 0:
                    continue
                for image in (
                    subprocess.check_output(
                        "ls "
                        + train_or_val_path
                        + datesync
                        + "/"
                        + "proj_depth/groundtruth/"
                        + imageLeftRight
                        + "/*.png",
                        shell=True,
                    )
                    .decode("utf-8")
                    .split("\n")
                ):
                    if len(image) == 0:
                        continue

                    label_path = (
                        train_or_val_path
                        + datesync
                        + "/"
                        + "proj_depth/groundtruth/"
                        + imageLeftRight
                        + "/"
                        + image[-14:]
                    )
                    # ~/UncertainDepths/data/external/KITTI/2011_09_30/2011_09_30_drive_0018_sync/image_02/data/0000000001.png
###'/home/frederik/UncertainDepths/data/external/KITTI/train/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/ 00000005.png'
### ls /home/frederik/UncertainDepths/data/external/KITTI/train/2011_09_26_drive_0001_sync/proj_depth/                   # /home/frederik/UncertainDepths/data/external/KITTI/train/2011_09_26_/2011_09_26_drive_0001_syncimage_02/data//home/frederik/UncertainDepths/data/external/KITTI/train/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02/0000000005.png,/home/frederik/UncertainDepths/data/external/KITTI/train/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02/home/frederik/UncertainDepths/data/external/KITTI/train/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02/0000000005.png

                    input_path = (
                        homepath
                        + datesync[:10]
                        + "/"
                        + datesync
                        + "/"
                        + imageLeftRight
                        + "/data/"
                        + image[-14:]
                    )
                    writer.writerow([input_path, label_path])
    return None


class KITTI_depth(Dataset):
    def __init__(
        self, data_dir: str, train_or_val: str = "train", transform=None, target_transform=None
    ) -> None:
        self.path_to_csv = (
            data_dir + train_or_val + "_input_and_label_paths.csv"
        )  # "/home/frederik/UncertainDepths/data/external/KITTI/train_input_and_label_paths.csv",
        extract_val_paths(data_dir, train_or_val, self.path_to_csv)
        self.transform = transform
        self.target_transform = target_transform
        self.paths_csv = pd.read_csv(self.path_to_csv)

    def __getitem__(self, idx):
        input_path = self.paths_csv.iloc[idx, 0]
        label_path = self.paths_csv.iloc[idx, 1]
        print(label_path)
        input_img = Image.open(input_path)
        label_img = Image.open(label_path)
        if self.transform:
            input_img = self.transform(input_img)
        if self.target_transform:
            label_img = self.target_transform(label_img)
        return input_img, label_img

    def __len__(self):
        return len(self.paths_csv)


class KITTIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/home/frederik/UncertainDepths/data/external/KITTI/",
        batch_size: int = 32,
        use_val_dir_for_val_and_test: bool = True,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.use_val_dir_for_val_and_test = use_val_dir_for_val_and_test
        self.transform = transforms.Compose([transforms.PILToTensor()])
        self.target_transform = transforms.Compose([transforms.PILToTensor()])
    def setup(self, stage: str) -> None:
        assert self.use_val_dir_for_val_and_test
        if stage == "fit":
            KITTI_train_val_set = KITTI_depth(data_dir = self.data_dir,train_or_val="train", transform = self.transform, target_transform =self.target_transform)
            self.KITTI_train_set = Subset(KITTI_train_val_set,np.arange(0,74272))#corresponds to 86% of dataset, while still being a different drive.
            self.KITTI_val_set = Subset(KITTI_train_val_set,np.arange(74272,len(KITTI_train_val_set)))
            plot_and_save_tensor_as_fig(self.KITTI_train_set[-1][0],"last_train_img")
            plot_and_save_tensor_as_fig(self.KITTI_val_set[0][0],"first_val_img")
        if stage == "test" or stage == "predict":
            self.KITTI_test_set = KITTI_depth(data_dir = self.data_dir,train_or_val="val", transform = self.transform, target_transform =self.target_transform)
            self.KITTI_predict_set = KITTI_depth(data_dir = self.data_dir,train_or_val="val", transform = self.transform, target_transform =self.target_transform)
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.KITTI_train_set, batch_size=self.batch_size)
 
if __name__ == "__main__":
    dataset = KITTI_depth(data_dir = "/home/frederik/UncertainDepths/data/external/KITTI/", train_or_val="train",transform = transforms.Compose([transforms.PILToTensor()]),target_transform = transforms.Compose([transforms.PILToTensor()]))
    a = iter(dataset)
    images = a.__next__()
    kitkat = KITTIDataModule(data_dir = "/home/frederik/UncertainDepths/data/external/KITTI/",batch_size=32)
    plot_and_save_tensor_as_fig(images[0],"atest")
    kitkat.setup("fit")
    
