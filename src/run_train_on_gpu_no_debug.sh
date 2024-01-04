#!/bin/bash

#SBATCH --output=slurm-folder/slurm_%j.out
#SBATCH --gres=gpu:a40:1 --time=3-00:00:00  --cpus-per-task=8 
conda activate UncertainDepths
python models/train_model.py in_debug=False save_images=True models=Online_Laplace neural_net_param_multiplication_factor=32 transforms.rand_aug=False trainer_args.max_epochs=10 models.prior_prec=500
