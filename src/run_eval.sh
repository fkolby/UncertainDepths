#!/bin/bash


#SBATCH --gres=gpu:a40:1 --time=0-12:00:00 --cpus-per-task=8 --output=slurm-folder/slurm_%j.out --mem-per-cpu=8192
#python models/run_eval.py 2024_01_24_08_44_52_7122_Online_Laplace
#python models/run_eval.py 2024_01_24_08_45_46_7124_Posthoc_Laplace
#python models/run_eval.py 2024_01_24_09_14_53_7127_Dropout
#python models/run_eval.py 2024_01_24_08_45_15_7123_Ensemble

#python models/run_eval.py 2024_02_05_10_20_16_1260_Online_Laplace


#python models/run_eval.py 2024_02_13_15_31_43_4766_Online_Laplace
#python models/run_eval.py 2024_02_10_08_18_37_3756_Posthoc_Laplace
python models/run_eval.py 2024_02_10_21_58_13_3840_Dropout
python models/run_eval.py 2024_02_10_08_18_53_3757_Ensemble