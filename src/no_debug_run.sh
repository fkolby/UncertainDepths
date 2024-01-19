#!/bin/bash

#SBATCH --output=slurm-folder/slurm_%j.out
#SBATCH --gres=gpu:a40:1 --time=3-00:00:00  --cpus-per-task=8 
python models/train_model.py in_debug=False save_images=True models=Online_Laplace neural_net_param_multiplication_factor=32 transforms.rand_aug=False trainer_args.max_epochs=50 models.hessian_memory_factor=0.999 models.update_hessian_probabilistically=True models.update_hessian_every=10 models.sample_last_n_epochs=50 models.dont_sample_parameters_during_training=False


