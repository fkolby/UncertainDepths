import lightning.pytorch as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms

from src.utility.train_utils import plot_and_save_tensor_as_fig, random_crop
from src.data.datasets.KITTI import KITTI_dataset

# from src.data.extract_paths_to_csv import extract_file_paths_to_csv
### inspired by adabins dataloading.


class base_datamodule(pl.LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig,
        use_val_dir_for_val_and_test: bool = True,
        transform=transforms.Compose([transforms.PILToTensor()]),
        target_transform=transforms.Compose([transforms.PILToTensor()]),
    ) -> None:
        super().__init__()
        self.use_val_dir_for_val_and_test = use_val_dir_for_val_and_test
        self.transform = transform
        self.target_transform = target_transform

        self.num_workers = cfg.dataset_params.num_workers
        self.batch_size = cfg.hyperparameters.batch_size
        self.input_height = cfg.dataset_params.input_height
        self.input_width = cfg.dataset_params.input_width
        self.cfg = cfg

    def setup(self, stage: str, **kwargs) -> None:
        raise NotImplementedError

    def prepare_data(self) -> None:
        print("setting up datamodule")

    def train_dataloader(self, shuffle=True) -> DataLoader:
        print("hi train_loader called")
        if self.train_set is None:
            raise NameError("Warning: train_set is not initialized. Did setup run successfully?")

        return DataLoader(
            self.train_set,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self, shuffle=False) -> DataLoader:
        print("val loading!")
        if self.val_set is None:
            raise NameError("Warning: train_set is not initialized. Did setup run successfully?")

        return DataLoader(
            self.val_set,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self, shuffle=False) -> DataLoader:
        print("test loading!")
        if self.test_set is None:
            raise NameError("Warning: train_set is not initialized. Did setup run successfully?")

        return DataLoader(
            self.test_set,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )


if __name__ == "__main__":
    dataset = KITTI_dataset(
        data_dir="/home/frederik/UncertainDepths/data/external/KITTI/",
        train_or_test="train",
        transform=transforms.Compose([transforms.PILToTensor()]),
        target_transform=transforms.Compose([transforms.PILToTensor()]),
        input_height=352,
        input_width=1216,
    )
    a = iter(dataset)
    images = a.__next__()
    plot_and_save_tensor_as_fig(images[0], "atest")
