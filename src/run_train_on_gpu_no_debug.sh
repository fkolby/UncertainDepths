#!/bin/bash

#SBATCH --gres=gpu:a40:1 --time=3-00:00:00  --cpus-per-task=8 
python models/train_model.py in_debug=False save_images=True models=Dropout neural_net_param_multiplication_factor=32 transforms.rand_aug=False
