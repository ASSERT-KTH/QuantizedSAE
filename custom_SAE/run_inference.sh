#!/bin/bash
#SBATCH --job-name=model_inference
#SBATCH --output=inference_output_%j.log
#SBATCH --error=inference_error_%j.log
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

python3 collect_model_states.py

