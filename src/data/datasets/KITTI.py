from src.data.datasets.base_dataset import depth_dataset
from PIL import Image
import os


class KITTI_dataset(depth_dataset):
    def __init__(self, *args, **kwargs):
        super.__init__(*args, **kwargs)
        self.cfg = kwargs.get("cfg")

        self.data_dir = self.cfg.dataset_params.data_dir
        self.path_to_file = (
            self.data_dir + "kitti_eigen_" + kwargs.get("train_or_test") + "_files.txt"
        )  # "/home/frederik/UncertainDepths/data/external/KITTI/train_input_and_label_paths.csv,
        # extract_file_paths_to_csv(data_dir, train_or_test, self.path_to_csv)

    def get_PIL_image(self, *args, **kwargs):
        sample_path = self.filenames[kwargs.pop("idx")]

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
        return input_img, label_img
