#!/bin/bash

# Set the target directory where you want to move the files
target_dir=~/UncertainDepths/data/external/KITTI/train_labels/

# Loop through each folder under ~/train
for folder in ~/UncertainDepths/data/external/KITTI/train/*; do
    # Extract the folder name placeholder from the path
    folder_name_placeholder=$(basename "$folder")

    # Define the directory containing image subdirectories
    img_subdir="$folder/proj_depth/groundtruth/"

    # Loop through each image folder (image_02 and image_03)
    for img_folder in "$img_subdir"image_02 "$img_subdir"image_03; do
        # Extract the image folder name (image_02 or image_03)
        img_folder_name=$(basename "$img_folder")

        # Loop through each PNG image in the current image folder
        for img_file in "$img_folder"/*.png; do
            # Extract the image file name from the path
            img_file_name=$(basename "$img_file")

            # Create the new name for the image file
            new_img_name="${folder_name_placeholder}_${img_folder_name}_${img_file_name}"

            # Move and rename the image file to the target directory
            mv "$img_file" "$target_dir/$new_img_name"
        done
    done
done


target_dir = ~/UncertainDepths/data/external/KITTI/all_pics_raw/





for folder in ~/UncertainDepths/data/external/KITTI/*2011_*/; do
    # Extract the folder name placeholder from the path
    folder_name_placeholder=$(basename "$folder")

    # Define the directory containing image subdirectories
    img_subdir="$folder/"2011_*_sync/image_0*/data/"

    # Loop through each image folder (image_02 and image_03)
    for img_folder in "$img_subdir"image_02 "$img_subdir"image_03; do
        # Extract the image folder name (image_02 or image_03)
        img_folder_name=$(basename "$img_folder")

        # Loop through each PNG image in the current image folder
        for img_file in "$img_folder"/*.png; do
            # Extract the image file name from the path
            img_file_name=$(basename "$img_file")

            # Create the new name for the image file
            new_img_name="${folder_name_placeholder}_${img_folder_name}_${img_file_name}"

            # Move and rename the image file to the target directory
            mv "$img_file" "$target_dir/$new_img_name"
        done
    done
done
