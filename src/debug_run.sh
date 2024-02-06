#!/bin/bash


#SBATCH --gres=gpu:a40:1 --time=03:00:00 --cpus-per-task=8 --output=slurm-folder/slurm_%j.out --mem-per-cpu=8192
python models/train_model.py dont_test_oom=False pdb_disabled=True save_images=True models=Ensemble neural_net_param_multiplication_factor=32 transforms.rand_aug=False 