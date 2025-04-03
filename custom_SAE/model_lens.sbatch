#!/bin/bash
#SBATCH --job-name=model_lens
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=16G
#SBATCH --time=00:30:00

python3 model_lens.py