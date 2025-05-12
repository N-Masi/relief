#!/bin/bash

#SBATCH --time=08:00:00
#SBATCH --mem=40G

# Specify a job name:
#SBATCH -J generate_landcover_240x121_pkls
#SBATCH -o %x.out



# Set up the environment by loading modules
module load cuda cudnn
module --ignore_cache load "conda"

# Run a script
conda init bash
conda activate faireenvconda
python sandbox/relief_loss_mvp.py
