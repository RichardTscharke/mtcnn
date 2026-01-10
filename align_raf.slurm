#!/bin/bash
set -e
#SBATCH --job-name=raf_align
#SBATCH --partition=NvidiaALL
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/raf_align_%j.out
#SBATCH --error=logs/raf_align_%j.err

# ====== Setup ======
source ~/.bashrc
conda activate mtcnn

echo "Job started on $(hostname)"
nvidia-smi

# ====== Dein Code ======
cd /home/t/tscharke/work/mtcnn
python preprocess_RAF.py
