#!/bin/bash

#SBATCH --gres=gpu:a40:1 --time=03:00:00 --cpus-per-task=8 
python models/train_model.py pdb_disabled=True save_images=True models=Ensemble neural_net_param_multiplication_factor=32 transforms.rand_aug=False