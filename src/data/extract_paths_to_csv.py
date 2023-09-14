import os
import csv
import subprocess


def extract_file_paths_to_csv(homepath: str, train_or_val: str, out_file: str) -> None:
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
