#!/bin/bash

#SBATCH --gres=gpu:a100:1 --time=1-00:00:00 --cpus-per-task=8 
python models/train_model.py pdb_disabled=True save_images=True models=Ensemble 