#!/bin/bash

#SBATCH -p gpu --gres=gpu:2 --gres-flags=enforce-binding

# Ensures all allocated cores are on the same node
#SBATCH -N 1

#SBATCH --time=08:00:00
#SBATCH --mem=40G

# Specify a job name:
#SBATCH -J 1979_2015_2_gpus_standard_loss_sfno_64x32
#SBATCH -o %x.out



# Set up the environment by loading modules
module load cuda cudnn
module --ignore_cache load "conda"

# Run a script
conda init bash
conda activate faireenvconda
torchrun --standalone --nnodes=1 --nproc_per_node=2 neuralops/sfno.py
