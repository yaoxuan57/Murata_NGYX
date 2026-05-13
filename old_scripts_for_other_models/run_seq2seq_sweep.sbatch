#!/bin/bash
#SBATCH -A prj0000000262
#SBATCH -p h200p
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -J seq2seq_sweep
#SBATCH -o ~/logs/%x_%j.out
#SBATCH -e ~/logs/%x_%j.err

set -euo pipefail

mkdir -p ~/logs

# Optional container path:
# module load apptainer
# apptainer exec --nv /path/to/container.sif python train_seq2seq_sweep.py --epochs 150

python -u train_seq2seq_sweep.py --epochs 150
