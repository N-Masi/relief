#!/bin/bash

# Request runtime (HH:MM:SS):
#SBATCH --time=07:00:00

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu --gres=gpu:1 --gres-flags=enforce-binding

# Default resources are 1 core with 2.8GB of memory.

#SBATCH --mem=140G

# Specify a job name:
#SBATCH -J test_neuralop_2mt_windspeed

# Specify an output file
#SBATCH -o test_neuralop_2mt_windspeed.out
#SBATCH -e test_neuralop_2mt_windspeed.out


# Set up the environment by loading modules
module load cuda cudnn
module load conda

# Run a script
conda init bash
conda activate faireenvconda
python -m neuralops.sfno
