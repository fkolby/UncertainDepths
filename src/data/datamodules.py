import lightning.pytorch as pl
import pdb
import numpy as np
from src.utility.train_utils import plot_and_save_tensor_as_fig
from PIL import Image
import pandas as pd
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import os
#from src.data.extract_paths_to_csv import extract_file_paths_to_csv
### inspired by adabins dataloading.

class KITTI_depth_dataset(Dataset):
    def __init__(
        self, data_dir: str, train_or_test: str = "train", transform=None, target_transform=None
    ) -> None:
        self.data_dir = data_dir
        self.path_to_file= (
            data_dir + "kitti_eigen_" + train_or_test + "_files.txt"
        )  # "/home/frederik/UncertainDepths/data/external/KITTI/train_input_and_label_paths.csv,
        #extract_file_paths_to_csv(data_dir, train_or_test, self.path_to_csv)
        self.transform = transform
        self.target_transform = target_transform
        if train_or_test in ["train", "test"]:
            with open(self.path_to_file, 'r') as f:
                self.filenames = f.readlines()
        else:
            Exception("Not implemented non-train/test for choosing files yet")
        self.train_or_test = train_or_test
    def __getitem__(self, idx):
        sample_path = self.filenames[idx]

        if self.train_or_test == 'train':
            if False: #I have not yet downloaded the eigen split for other camera. random.random()>0.5 #self.args.dataset == 'kitti' and self.args.use_right is True and random.random() > 0.5:
                input_path = os.path.join(self.args.data_path, self.data_dir + sample_path.split()[3])
                label_path = os.path.join(self.args.gt_path, self.data_dir + sample_path.split()[4])
            else:
                input_path = os.path.join(self.data_dir, sample_path.split()[0])
                label_path = os.path.join(self.data_dir, "train", sample_path.split()[1])
        else:
            Exception("Not implemented get-item in dataloader for test yet.")
        #input_path = self.paths_csv.iloc[idx, 0]
        #label_path = self.paths_csv.iloc[idx, 1]


        input_img = Image.open(input_path)
        label_img = Image.open(label_path)
        if self.transform:
            input_img = self.transform(input_img).float()
        if self.target_transform:
            label_img = self.target_transform(label_img).float()
        return input_img, label_img

    def __len__(self):
        return len(self.filenames)


class KITTIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/home/frederik/UncertainDepths/data/external/KITTI/",
        batch_size: int = 32,
        use_val_dir_for_val_and_test: bool = True,
        transform=transforms.Compose([transforms.PILToTensor()]),
        target_transform=transforms.Compose([transforms.PILToTensor()]),
        num_workers: int = 8,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.use_val_dir_for_val_and_test = use_val_dir_for_val_and_test
        self.transform = transform
        self.target_transform = target_transform
        self.KITTI_val_set = None
        self.KITTI_train_set = None
        self.KITTI_test_set = None
        self.KITTI_predict_set = None
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        print("setting up datamodule")

    def setup(self, stage: str) -> None:
        print(stage)
        assert self.use_val_dir_for_val_and_test
        if stage == "fit":
            print("got to fittingsss")
            KITTI_train_val_set = KITTI_depth_dataset(
                data_dir=self.data_dir,
                train_or_test="train",
                transform=self.transform,
                target_transform=self.target_transform,
            )
            print("got to evaluating KITTI train val set")
            self.KITTI_train_set = Subset(
                #KITTI_train_val_set, np.arange(0, 74272, dtype=int).tolist() use only if full dataset (not eigen)
                KITTI_train_val_set, np.arange(0, 18525, dtype=int).tolist()
            )  # corresponds to 86% of dataset, while still being a different drive.
            self.KITTI_val_set = Subset(
                KITTI_train_val_set, np.arange(18525, len(KITTI_train_val_set), dtype=int).tolist()
            )
            plot_and_save_tensor_as_fig(self.KITTI_train_set[-1][0], "last_train_img")
            plot_and_save_tensor_as_fig(self.KITTI_val_set[0][0], "first_val_img")
            assert(self.KITTI_train_set[-1][0].shape == self.KITTI_val_set[0][0].shape)
            
            print("got to last part of if fit")
        if stage == "test" or stage == "predict":
            self.KITTI_test_set = KITTI_depth_dataset(
                data_dir=self.data_dir,
                train_or_test="val",
                transform=self.transform,
                target_transform=self.target_transform,
            )
            self.KITTI_predict_set = KITTI_depth_dataset(
                data_dir=self.data_dir,
                train_or_test="val",
                transform=self.transform,
                target_transform=self.target_transform,
            )

    def train_dataloader(self) -> DataLoader:
        print("hi train_loader called")
        if self.KITTI_train_set is None:
            raise NameError(
                "Warning: KITTI_train_set is not initialized. Did setup run successfully?"
            )

        return DataLoader(
            self.KITTI_train_set, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        print("val loading!")
        if self.KITTI_val_set is None:
            raise NameError(
                "Warning: KITTI_train_set is not initialized. Did setup run successfully?"
            )


        return DataLoader(
            self.KITTI_val_set, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.KITTI_test_set, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.KITTI_predict_set, batch_size=self.batch_size, num_workers=self.num_workers
        )


if __name__ == "__main__":
    dataset = KITTI_depth_dataset(
        data_dir="/home/frederik/UncertainDepths/data/external/KITTI/",
        train_or_test="train",
        transform=transforms.Compose([transforms.PILToTensor()]),
        target_transform=transforms.Compose([transforms.PILToTensor()]),
    )
    a = iter(dataset)
    images = a.__next__()
    kitkat = KITTIDataModule(
        data_dir="/home/frederik/UncertainDepths/data/external/KITTI/", batch_size=32
    )
    plot_and_save_tensor_as_fig(images[0], "atest")
    kitkat.setup("fit")
