#!/bin/bash

#SBATCH --time=10:00:00
#SBATCH --mem=40G

# Specify a job name:
#SBATCH -J generate_gdf_group_data
#SBATCH -o %x.out



# Set up the environment by loading modules
module load cuda cudnn
module --ignore_cache load "conda"

# Run a script
conda init bash
conda activate faireenvconda
python sandbox/rmse_gdf.py
