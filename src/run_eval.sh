#!/bin/bash


#SBATCH --gres=gpu:a40:1 --time=4-00:00:00 --cpus-per-task=8 --output=slurm-folder/slurm_%j.out --mem-per-cpu=8192
#python models/run_eval.py 2024_01_24_08_44_52_7122_Online_Laplace
#python models/run_eval.py 2024_01_24_08_45_46_7124_Posthoc_Laplace
#python models/run_eval.py 2024_01_24_09_14_53_7127_Dropout
#python models/run_eval.py 2024_01_24_08_45_15_7123_Ensemble

python models/run_eval.py 2024_02_05_10_20_16_1260_Online_Laplace