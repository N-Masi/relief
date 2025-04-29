#!/bin/bash

#SBATCH --partition=gpu-he
#SBATCH --constraint=h100

# Ensures all allocated cores are on the same node
#SBATCH -N 1

#SBATCH --time=10:00:00

# Specify a job name:
#SBATCH -J full_loss_test_sfno
#SBATCH -o %j.out

# Set up the environment by loading modules
module load cuda cudnn
module load conda

# Run a script
conda init bash
conda activate faireenvconda
python -m neuralops.sfno
