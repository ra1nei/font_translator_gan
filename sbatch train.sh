#!/bin/bash
#SBATCH -o logs/phase1_%j.out
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

# Cập nhật code
git pull origin main

# Load biến môi trường từ ~/.bashrc (WANDB_API_KEY sẽ có ở đây)
source ~/.bashrc

# Vào project
cd ~/data/TDKD/font_translator_gan || exit

# Load conda env
source /data/cndt_hangdv/miniconda3/bin/activate fontgan || { echo "Failed to activate conda env"; exit 1; }

# Đảm bảo wandb login với đúng key của bạn
wandb login --relogin $WANDB_API_KEY

# Chạy training
bash "./train.sh"