#!/bin/bash

#SBATCH --gres=gpu:1 --time=3-00:00:00  --cpus-per-task=8 
python models/train_model.py in_debug=False save_images=True