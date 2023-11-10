#!/bin/bash

#SBATCH --gres=gpu:1 --time=00:05:00  
python models/train_model.py in_debug=False