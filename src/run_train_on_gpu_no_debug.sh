#!/bin/bash

#SBATCH --gres=gpu:a100:1 --time=03:00:00  --cpus-per-task=8 
python models/train_model.py in_debug=False save_images=True models=stochastic_unet