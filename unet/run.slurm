#!/bin/bash
#SBATCH --job-name=xview2_unet
#SBATCH -A sds_ag
#SBATCH -p gpu
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu

# Load environment
module purge
module load miniforge

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/kyn5du/.conda/envs/nn_env

# Move to working directory
cd /home/kyn5du/autoresearch

python program.py

python evaluate.py
