#!/bin/bash

#SBATCH --gres=gpu:1 --time=1-00:00:00  
python models/train_model.py in_debug=False