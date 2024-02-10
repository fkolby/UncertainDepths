from torch.utils.data import Dataset
from PIL import Image
from numpy import random
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


class depth_dataset(Dataset):
    def __init__(
        self,
        cfg: DictConfig,
        train_or_test: str = "train",
        train_or_test_transform: str = "train",
        transform=None,
        target_transform=None,
        **kwargs,
    ) -> None:
        self.transform = transform
        self.target_transform = target_transform

        self.train_or_test = train_or_test
        self.train_or_test_transform = train_or_test_transform
        self.input_height = cfg.dataset_params.input_height
        self.input_width = cfg.dataset_params.input_width
        self.dataset_type_is_ood = kwargs.get("dataset_type_is_ood", False)
        self.cfg = cfg
        print("Kwargs")
        print(kwargs)
        print(self.dataset_type_is_ood)
        print(cfg.OOD.use_white_noise_box_test)

    def get_PIL_image(self, dataset_type="train", *args, **kwargs):
        # should return input_img, label_img
        raise NotImplementedError

    def __getitem__(self, idx):
        input_img, label_img = self.get_PIL_image(idx=idx)
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

            pink = torch.tensor((207, 17, 191)).reshape(3,1,1)

            box_y_offset = 100
            box_x_offset = 100

            box_y_start = np.random.randint(0, h - box_y_offset)
            box_x_start = np.random.randint(0, w - 100)
            input_img[
                :,
                box_y_start : box_y_start + box_y_offset,
                box_x_start : box_x_start + box_x_offset,
            ] = torch.ones((3, box_y_offset, box_x_offset))*pink/255

            OOD_class = torch.zeros_like(label_img, dtype=torch.int16)  # 0 is in distribution
            OOD_class[
                :,
                box_y_start : box_y_start + box_y_offset,
                box_x_start : box_x_start + box_x_offset,
            ] = 1

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
