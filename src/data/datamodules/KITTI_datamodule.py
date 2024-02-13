from src.data.datamodules.base_datamodule import base_datamodule
from src.data.datasets.KITTI import KITTI_dataset
import numpy as np
from torch.utils.data import Subset


class KITTI_datamodule(base_datamodule):
    """Just a container for running setup - which just setups dataloaders on the KITTI dataset class"""
    def __init__(self, *args, **kwargs):
        self.data_dir = kwargs.pop(
            "data_dir", "/home/frederik/UncertainDepths/data/external/KITTI/"
        )
        super().__init__(*args, **kwargs)

    def setup(self, stage: str, **kwargs) -> None:
        print(stage)
        assert self.use_val_dir_for_val_and_test
        if stage == "fit":
            print("got to fittingsss")
            train_val_set = KITTI_dataset(
                train_or_test="train",
                transform=self.transform,
                target_transform=self.target_transform,
                cfg=self.cfg,
                **kwargs,
            )
            if not self.cfg.dataset_params.test_set_is_actually_valset:
                print("Currently testset is just valset without disturbing augmentations.")
                raise NotImplementedError
            else:
                test_set = KITTI_dataset(
                    train_or_test="train",
                    train_or_test_transform="val",
                    transform=self.transform,
                    target_transform=self.target_transform,
                    cfg=self.cfg,
                    **kwargs,
                )
                print("got to evaluating KITTI train val set")
                self.train_set = Subset(
                    # train_val_set, np.arange(0, 74272, dtype=int).tolist() use only if full dataset (not eigen)
                    train_val_set,
                    np.arange(0, 16525, dtype=int).tolist(),
                )  # corresponds to 86% of dataset, while still being a different drive.

                self.val_set = Subset(train_val_set, np.arange(16525, 18525, dtype=int).tolist())
                print("len of train-vallength: ", len(train_val_set))

                self.test_set = Subset(
                    test_set, np.arange(18525, len(test_set), dtype=int).tolist()
                )
                # plot_and_save_tensor_as_fig(self.train_set[-1][0], "last_train_img")
                # plot_and_save_tensor_as_fig(self.val_set[0][0], "first_val_img")
                assert self.train_set[-1][0].shape == self.val_set[0][0].shape

                print("got to last part of if fit")
        if stage == "test" or stage == "predict":
            raise NotImplementedError
