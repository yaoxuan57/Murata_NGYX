#!/bin/bash
#SBATCH -A prj0000000262
#SBATCH -p h200p
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -J mcdlinear_576_36
#SBATCH -o ~/logs/%x_%j.out
#SBATCH -e ~/logs/%x_%j.err

set -euo pipefail

mkdir -p ~/logs

# Optional container path:
# module load apptainer
# apptainer exec --nv /path/to/container.sif python -u train_multicomponent_dlinear_sweep.py --input-lens 576 --pred-lens 36

python -u train_multicomponent_dlinear_sweep.py \
  --output-dir outputs_multicomponent_dlinear_576_to_36 \
  --input-lens 576 \
  --pred-lens 36 \
  --epochs 250 \
  --batch-size 16 \
  --lr 5e-4 \
  --weight-decay 1e-6 \
  --early-stopping-patience 20 \
  --scheduler-patience 6 \
  --k-slow 97 \
  --k-mid 33 \
  --k-fast 9 \
  --loss-point-weight 0.7 \
  --loss-diff-weight 0.9 \
  --loss-curvature-weight 0.5 \
  --loss-variance-weight 0.2 \
  --save-window-plots
