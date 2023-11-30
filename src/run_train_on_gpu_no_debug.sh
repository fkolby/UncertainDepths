#!/bin/bash

#SBATCH --gres=gpu:a100:1 --time=01:00:00  --cpus-per-task=8 
python models/train_model.py in_debug=False save_images=True models=stochastic_unet +trainer_args.limit_train_batches=0.01 +trainer_args.limit_val_batches=0.01 +trainer_args.fast_dev_run=True