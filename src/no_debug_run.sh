#!/bin/bash

#SBATCH --output=slurm-folder/slurm_%j.out
#SBATCH --gres=gpu:a100:1 --time=1-12:00:00  --cpus-per-task=8 --mem-per-cpu=8G
#python models/train_model.py in_debug=False save_images=True models=Posthoc_Laplace neural_net_param_multiplication_factor=32 hyperparameters.learning_rate=3e-4 transforms.rand_aug=False trainer_args.max_epochs=150 
#python models/train_model.py models=Dropout in_debug=False save_images=True neural_net_param_multiplication_factor=32 hyperparameters.learning_rate=3e-4 transforms.rand_aug=False trainer_args.max_epochs=50 #models.hessian_memory_factor=0.9999 models.hessian_scale=1e+4 models.update_hessian_probabilistically=True models.update_hessian_every=10 models.sample_last_n_epochs=50 models.dont_sample_parameters_during_training=False models.use_exp_average_instead=False
#python models/train_model.py models=Ensemble in_debug=False save_images=True neural_net_param_multiplication_factor=32 hyperparameters.learning_rate=3e-4 transforms.rand_aug=False trainer_args.max_epochs=100 #models.hessian_memory_factor=0.9999 models.hessian_scale=1e+4 models.update_hessian_probabilistically=True models.update_hessian_every=10 models.sample_last_n_epochs=50 models.dont_sample_parameters_during_training=False models.use_exp_average_instead=False

python models/train_model.py models.sample_last_n_epochs=0 hyperparameters.batch_size=16 models.hessian_initial_multiplication_factor=1e+4 models=Online_Laplace in_debug=False save_images=True neural_net_param_multiplication_factor=32 hyperparameters.learning_rate=3e-4 transforms.rand_aug=False trainer_args.max_epochs=150 models.hessian_memory_factor=0.9999 models.hessian_scale=1e+4 models.update_hessian_probabilistically=True models.update_hessian_every=10  models.dont_sample_parameters_during_training=False models.use_exp_average_instead=False 





