#!/bin/bash
#SBATCH -A prj0000000262
#SBATCH -p h200p
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -J mambaformer_tuning
#SBATCH -o ~/logs/%x_%j.out
#SBATCH -e ~/logs/%x_%j.err

set -euo pipefail

mkdir -p ~/logs

ROOT=outputs_mambaformer_tuning
RUNS=${ROOT}/runs
BEST=${ROOT}/best_model
mkdir -p "${RUNS}" "${BEST}"

COMMON_TRAIN="--epochs 250 --batch-size 16 --early-stopping-patience 20 --scheduler-patience 6 --pred-lens 288"

python -u train_mambaformer_sweep.py ${COMMON_TRAIN} --input-lens 864 --d-model 128 --d-inner 256 --nhead 8 --num-mamba-layers 3 --num-former-layers 2 --dim-feedforward 256 --lr 5e-4 --weight-decay 0 --output-dir "${RUNS}/cfg_864_d128_i256_m3_f2_lr5e4"
python -u train_mambaformer_sweep.py ${COMMON_TRAIN} --input-lens 1296 --d-model 128 --d-inner 256 --nhead 8 --num-mamba-layers 4 --num-former-layers 2 --dim-feedforward 256 --lr 5e-4 --weight-decay 0 --output-dir "${RUNS}/cfg_1296_d128_i256_m4_f2_lr5e4"
python -u train_mambaformer_sweep.py ${COMMON_TRAIN} --input-lens 1728 --d-model 160 --d-inner 320 --nhead 8 --num-mamba-layers 4 --num-former-layers 3 --dim-feedforward 320 --lr 2e-4 --weight-decay 1e-6 --output-dir "${RUNS}/cfg_1728_d160_i320_m4_f3_lr2e4"
python -u train_mambaformer_sweep.py ${COMMON_TRAIN} --input-lens 2160 --d-model 192 --d-inner 384 --nhead 8 --num-mamba-layers 5 --num-former-layers 3 --dim-feedforward 384 --lr 2e-4 --weight-decay 1e-6 --output-dir "${RUNS}/cfg_2160_d192_i384_m5_f3_lr2e4"

python -u select_best_sweep_run.py --runs-root "${RUNS}" --out-json "best_run_selection.json"

python -u rerun_best_sweep_with_plots.py \
  --selection-json "${RUNS}/best_run_selection.json" \
  --train-script "train_mambaformer_sweep.py" \
  --output-root "${BEST}" \
  ${COMMON_TRAIN}

python -u select_best_sweep_run.py --runs-root "${BEST}" --out-json "best_model_selection.json"

echo "MambaFormer tuning completed."
echo "Candidates ranking: ${RUNS}/sweep_run_ranking.csv"
echo "Selected best candidate metadata: ${RUNS}/best_run_selection.json"
echo "Candidate outputs under: ${RUNS}/<cfg>/rolling_window_forecasts/plots"
echo "Best-model outputs under: ${BEST}/<best_cfg>/rolling_window_forecasts/plots"
