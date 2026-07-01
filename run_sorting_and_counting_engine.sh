#!/bin/bash
#SBATCH --job-name=bat_pipeline_master
#SBATCH --output=master_pipeline_%j.log
#SBATCH --error=master_pipeline_%j.err
#SBATCH --time=12:00:00               
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8             
#SBATCH --gpus-per-node=1             

# --- UC ARC SPECIFIC CHANGES ---
#SBATCH --account=YOUR_UC_PROJECT_ID   # Replace with UC billing account
#SBATCH --partition=gpu                # Replace with the exact UC GPU partition name

echo "Master sorting and extraction pipeline started on $(date)"

# Load the exact CUDA module name UC uses (Run 'module avail cuda' to find it)
module load cuda/xx.x.x  

# Activate your environment
source activate bat_env

# Execute the processing engine
python 	count_unique_bats.py

echo "Bats sorted on $(date)"