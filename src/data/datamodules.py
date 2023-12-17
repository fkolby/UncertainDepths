import os
import pdb
import random

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from src.utility.train_utils import plot_and_save_tensor_as_fig, random_crop

# from src.data.extract_paths_to_csv import extract_file_paths_to_csv
### inspired by adabins dataloading.


class KITTI_depth_dataset(Dataset):
    def __init__(
        self,
        cfg: DictConfig,
        train_or_test: str = "train",
        train_or_test_transform: str = "train",
        transform=None,
        target_transform=None,
        **kwargs,
    ) -> None:
        self.data_dir = cfg.dataset_params.data_dir
        self.path_to_file = (
            self.data_dir + "kitti_eigen_" + train_or_test + "_files.txt"
        )  # "/home/frederik/UncertainDepths/data/external/KITTI/train_input_and_label_paths.csv,
        # extract_file_paths_to_csv(data_dir, train_or_test, self.path_to_csv)
        self.transform = transform
        self.target_transform = target_transform
        if train_or_test in ["train", "test"]:
            with open(self.path_to_file, "r") as f:
                self.filenames = f.readlines()
        else:
            Exception("Not implemented non-train/test for choosing files yet")
        self.train_or_test = train_or_test
        self.train_or_test_transform = train_or_test_transform
        self.input_height = cfg.dataset_params.input_height
        self.input_width = cfg.dataset_params.input_width
        print(kwargs)
        self.dataset_type_is_ood = kwargs.get("dataset_type_is_ood", False)
        self.cfg = cfg

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]

        if self.train_or_test == "train":
            if (
                False
            ):  # I have not yet downloaded the eigen split for other camera. random.random()>0.5 #self.args.dataset == 'kitti' and self.args.use_right is True and random.random() > 0.5:
                input_path = os.path.join(
                    self.args.data_path, self.data_dir + sample_path.split()[3]
                )
                label_path = os.path.join(self.args.gt_path, self.data_dir + sample_path.split()[4])
            else:
                input_path = os.path.join(self.data_dir, sample_path.split()[0])
                label_path = os.path.join(self.data_dir, "train", sample_path.split()[1])
        else:
            Exception("Not implemented get-item in dataloader for test yet.")
        # input_path = self.paths_csv.iloc[idx, 0]
        # label_path = self.paths_csv.iloc[idx, 1]

        input_img = Image.open(input_path)
        label_img = Image.open(label_path)

        """ 

                no_transforms = transforms.Compose(
                    [
                        transforms.PILToTensor(),
                        transforms.CenterCrop((352, 1216)),
                        transforms.Lambda(lambda x: x / 256),  # 256 as per devkit
                    ]
                )
        label_untransformed = no_transforms(label_img.copy()).detach()  # Kitti benchmark crop.
        """

        if self.cfg.transforms.kb_crop:  # region-of-interest-crop
            height = input_img.height
            width = input_img.width
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            label_img = label_img.crop(
                (left_margin, top_margin, left_margin + 1216, top_margin + 352)
            )
            input_img = input_img.crop(
                (left_margin, top_margin, left_margin + 1216, top_margin + 352)
            )

        if self.cfg.transforms.rotate and self.train_or_test_transform == "train":
            random_angle = (
                (random.random() - 0.5) * 2 * self.cfg.transforms.rotational_degree
            )  # dont set seed here, as that must be in trainer, for ensuring consistency for ensembles.
            input_img = self.rotate_image(input_img, random_angle)
            label_img = self.rotate_image(label_img, random_angle, flag=Image.NEAREST)

        if self.transform:
            input_img = self.transform(input_img).float()
        if self.target_transform:
            label_img = self.target_transform(label_img).float()

        if self.cfg.transforms.flip_LR and self.train_or_test_transform == "train":
            assert isinstance(input_img, torch.Tensor)
            if random.random() > 0.5:
                input_img = torch.flip(input_img, dims=[-1])  # CxHxW
                label_img = torch.flip(label_img, dims=[-1])  # CxHxW

        if self.cfg.transforms.rand_aug and self.train_or_test_transform == "train":
            if random.random() > 0.5:
                input_img = self.augment_image(input_img)

        if self.cfg.transforms.rand_crop and self.train_or_test_transform == "train":
            input_img, label_img = self.random_crop(
                img=input_img, depth=label_img, height=self.input_height, width=self.input_width
            )

        if self.dataset_type_is_ood and self.cfg.OOD.use_white_noise_box_test:
            c, h, w = input_img.shape

            box_y_start = torch.randint(0, h - 50, ((1)))
            box_x_start = torch.randint(0, w - 50, ((1)))
            input_img[
                :, box_y_start : box_y_start + 50, box_x_start : box_x_start + 50
            ] = torch.rand(3, 50, 50)

            OOD_class = torch.zeros_like(label_img)  # 0 is in distribution
            OOD_class[box_y_start : box_y_start + 50, box_x_start : box_x_start + 50] = 1

            return input_img, label_img, OOD_class

        return (
            input_img,
            label_img,
            torch.zeros_like(label_img),
        )  # last output is only used as placeholder (0-class is in-distribution.)

    def rotate_image(
        self, image: Image.Image, angle, flag=Image.BILINEAR
    ):  # taken from ZoeDepth-https://github.com/isl-org/ZoeDepth/blob/edb6daf45458569e24f50250ef1ed08c015f17a7/zoedepth/data/data_mono.py#L70
        result = image.rotate(angle, resample=flag)
        return result

    def augment_image(
        self, image: torch.Tensor
    ):  # inspired by ZoeDepth (https://github.com/isl-org/ZoeDepth/blob/edb6daf45458569e24f50250ef1ed08c015f17a7/zoedepth/data/data_mono.py#L484)
        # gamma augmentation
        assert isinstance(image, torch.Tensor)
        assert torch.min(image).item() < -0.05  # check to ensure normalization has been done.
        # denormalize (necessary to exponentiate, as negative numbers are not allowed.):
        mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        if len(image.shape) == 3:
            image = image * std + mean
        else:
            raise ValueError

        image = torch.clamp(image, 0, 1)

        gamma = random.uniform(0.9, 1.1)
        image_aug = image**gamma

        # brightness augmentation
        if self.cfg.dataset_params.name == "nyu":
            brightness = random.uniform(0.75, 1.25)
            var_bright = 1 / 12 * (1.25 - 0.75) ** 2
        else:
            brightness = random.uniform(0.9, 1.1)
            var_bright = 1 / 12 * (1.1 - 0.9) ** 2
        image_aug = image_aug * brightness

        # color augmentation
        colors = torch.rand(size=(3, 1, 1)) * 0.2 + 0.9  # 0.9-1.1
        var_col = 1 / 12 * (1.1 - 0.9) ** 2
        white = torch.ones((3, image.shape[1], image.shape[2]))
        color_image = colors * white
        image_aug *= color_image
        # image_aug = torch.clamp(image_aug, 0, 1)

        var_aug = (var_bright + 1) * (var_col + 1) * (
            (1.042017) * std**2 + 0.99964 * mean**2
        ) - 1 * 1 * (
            mean**2 * 0.99964
        )  # variance of augmentation (apart from gamma)
        # constants comes from sd of result of simulation of 10000 rnorm(0.4850,0.229) clamped at (0,1) gamma transformed by 10000 runif(0.9,1.1) variables, relative to moments of rnorm-variable)
        # renormalize
        if len(image_aug.shape) == 3:
            out = (image_aug - mean) / (
                var_aug ** (1 / 2)
            )  # make sure it is still whitened requires at dividing final variance by std_bright
        else:
            raise ValueError

        assert not ((torch.sum(torch.isnan(image)) > 0).item())
        return out

    def __len__(self):
        return len(self.filenames)


class KITTIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig,
        data_dir: str = "/home/frederik/UncertainDepths/data/external/KITTI/",
        batch_size: int = 32,
        use_val_dir_for_val_and_test: bool = True,
        transform=transforms.Compose([transforms.PILToTensor()]),
        target_transform=transforms.Compose([transforms.PILToTensor()]),
        num_workers: int = 8,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.use_val_dir_for_val_and_test = use_val_dir_for_val_and_test
        self.transform = transform
        self.target_transform = target_transform
        self.KITTI_val_set = None
        self.KITTI_train_set = None
        self.KITTI_test_set = None
        self.KITTI_predict_set = None
        self.num_workers = cfg.dataset_params.num_workers
        self.batch_size = cfg.hyperparameters.batch_size
        self.input_height = cfg.dataset_params.input_height
        self.input_width = cfg.dataset_params.input_width
        self.cfg = cfg

    def prepare_data(self) -> None:
        print("setting up datamodule")

    def setup(self, stage: str, **kwargs) -> None:
        print(stage)
        assert self.use_val_dir_for_val_and_test
        if stage == "fit":
            print("got to fittingsss")
            KITTI_train_val_set = KITTI_depth_dataset(
                train_or_test="train",
                transform=self.transform,
                target_transform=self.target_transform,
                cfg=self.cfg,
                kwargs=kwargs,
            )
            if not self.cfg.dataset_params.test_set_is_actually_valset:
                print("Currently testset is just valset without disturbing augmentations.")
                raise NotImplementedError
            KITTI_test_set = KITTI_depth_dataset(
                train_or_test="train",
                train_or_test_transform="val",
                transform=self.transform,
                target_transform=self.target_transform,
                cfg=self.cfg,
                kwargs=kwargs,
            )
            print("got to evaluating KITTI train val set")
            self.KITTI_train_set = Subset(
                # KITTI_train_val_set, np.arange(0, 74272, dtype=int).tolist() use only if full dataset (not eigen)
                KITTI_train_val_set,
                np.arange(0, 18525, dtype=int).tolist(),
            )  # corresponds to 86% of dataset, while still being a different drive.
            self.KITTI_val_set = Subset(
                KITTI_train_val_set, np.arange(18525, len(KITTI_train_val_set), dtype=int).tolist()
            )
            print("len of train-vallength: ", len(KITTI_train_val_set))

            self.KITTI_test_set = Subset(
                KITTI_test_set, np.arange(18525, len(KITTI_test_set), dtype=int).tolist()
            )
            # plot_and_save_tensor_as_fig(self.KITTI_train_set[-1][0], "last_train_img")
            # plot_and_save_tensor_as_fig(self.KITTI_val_set[0][0], "first_val_img")
            assert self.KITTI_train_set[-1][0].shape == self.KITTI_val_set[0][0].shape

            print("got to last part of if fit")
        if stage == "test" or stage == "predict":
            raise NotImplementedError

    def train_dataloader(self, shuffle=True) -> DataLoader:
        print("hi train_loader called")
        if self.KITTI_train_set is None:
            raise NameError(
                "Warning: KITTI_train_set is not initialized. Did setup run successfully?"
            )

        return DataLoader(
            self.KITTI_train_set,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self, shuffle=False) -> DataLoader:
        print("val loading!")
        if self.KITTI_val_set is None:
            raise NameError(
                "Warning: KITTI_train_set is not initialized. Did setup run successfully?"
            )

        return DataLoader(
            self.KITTI_val_set,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self, shuffle=False) -> DataLoader:
        print("test loading!")
        if self.KITTI_test_set is None:
            raise NameError(
                "Warning: KITTI_train_set is not initialized. Did setup run successfully?"
            )

        return DataLoader(
            self.KITTI_test_set,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )


if __name__ == "__main__":
    dataset = KITTI_depth_dataset(
        data_dir="/home/frederik/UncertainDepths/data/external/KITTI/",
        train_or_test="train",
        transform=transforms.Compose([transforms.PILToTensor()]),
        target_transform=transforms.Compose([transforms.PILToTensor()]),
        input_height=352,
        input_width=1216,
    )
    a = iter(dataset)
    images = a.__next__()
    kitkat = KITTIDataModule(
        data_dir="/home/frederik/UncertainDepths/data/external/KITTI/", batch_size=32
    )
    plot_and_save_tensor_as_fig(images[0], "atest")
    kitkat.setup("fit")
