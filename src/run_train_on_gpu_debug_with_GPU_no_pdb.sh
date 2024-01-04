#!/bin/bash


#SBATCH --gres=gpu:1 --time=03:00:00 --cpus-per-task=8 --output=slurm-folder/slurm_%j.out
python models/train_model.py pdb_disabled=True save_images=True models=Online_Laplace neural_net_param_multiplication_factor=32 transforms.rand_aug=False 