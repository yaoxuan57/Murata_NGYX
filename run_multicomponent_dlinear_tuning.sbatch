#!/bin/bash
#SBATCH -A prj0000000262
#SBATCH -p h200p
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -J multicomponent_dlinear_tuning
#SBATCH -o ~/logs/%x_%j.out
#SBATCH -e ~/logs/%x_%j.err

set -euo pipefail

mkdir -p ~/logs

ROOT=outputs_multicomponent_dlinear_tuning
RUNS=${ROOT}/runs
BEST=${ROOT}/best_model
mkdir -p "${RUNS}" "${BEST}"

COMMON_TRAIN="--epochs 250 --batch-size 16 --early-stopping-patience 20 --scheduler-patience 6 --pred-lens 288"
COMMON_LOSS="--loss-point-weight 0.7 --loss-diff-weight 0.9 --loss-curvature-weight 0.5 --loss-variance-weight 0.2"

python -u train_multicomponent_dlinear_sweep.py ${COMMON_TRAIN} ${COMMON_LOSS} --save-window-plots --input-lens 864 --k-slow 97 --k-mid 33 --k-fast 9 --lr 5e-4 --weight-decay 0 --output-dir "${RUNS}/cfg_864_k97_33_9_lr5e4"
python -u train_multicomponent_dlinear_sweep.py ${COMMON_TRAIN} ${COMMON_LOSS} --save-window-plots --input-lens 1296 --k-slow 129 --k-mid 41 --k-fast 11 --lr 5e-4 --weight-decay 0 --output-dir "${RUNS}/cfg_1296_k129_41_11_lr5e4"
python -u train_multicomponent_dlinear_sweep.py ${COMMON_TRAIN} ${COMMON_LOSS} --save-window-plots --input-lens 1728 --k-slow 161 --k-mid 49 --k-fast 13 --lr 2e-4 --weight-decay 1e-6 --output-dir "${RUNS}/cfg_1728_k161_49_13_lr2e4"
python -u train_multicomponent_dlinear_sweep.py ${COMMON_TRAIN} ${COMMON_LOSS} --save-window-plots --input-lens 2160 --k-slow 193 --k-mid 65 --k-fast 17 --lr 2e-4 --weight-decay 1e-6 --output-dir "${RUNS}/cfg_2160_k193_65_17_lr2e4"

python -u select_best_sweep_run.py --runs-root "${RUNS}" --out-json "best_run_selection.json"

python -u rerun_best_sweep_with_plots.py \
  --selection-json "${RUNS}/best_run_selection.json" \
  --train-script "train_multicomponent_dlinear_sweep.py" \
  --output-root "${BEST}" \
  ${COMMON_TRAIN} \
  ${COMMON_LOSS}

python -u select_best_sweep_run.py --runs-root "${BEST}" --out-json "best_model_selection.json"

echo "Multi-component DLinear tuning completed."
echo "Candidates ranking: ${RUNS}/sweep_run_ranking.csv"
echo "Selected best candidate metadata: ${RUNS}/best_run_selection.json"
echo "Candidate run plots under: ${RUNS}/<cfg>/rolling_window_forecasts/plots"
echo "Best-model-only window plots under: ${BEST}/<best_cfg>/rolling_window_forecasts/plots"
