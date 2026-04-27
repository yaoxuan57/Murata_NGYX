#!/bin/bash
#SBATCH -A prj0000000262
#SBATCH -p h200p
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -J dlinear_tuning
#SBATCH -o ~/logs/%x_%j.out
#SBATCH -e ~/logs/%x_%j.err

set -euo pipefail

mkdir -p ~/logs

ROOT=outputs_dlinear_tuning
RUNS=${ROOT}/runs
BEST=${ROOT}/best_model
mkdir -p "${RUNS}" "${BEST}"

COMMON_TRAIN="--epochs 250 --batch-size 16 --early-stopping-patience 20 --scheduler-patience 6 --pred-lens 36"
COMMON_LOSS="--loss-point-weight 0.7 --loss-diff-weight 0.9 --loss-curvature-weight 0.5 --loss-variance-weight 0.2"

# 1) Run candidate configs without heavy per-window plot generation.
python -u train_dlinear_sweep.py ${COMMON_TRAIN} ${COMMON_LOSS} --no-window-plots --input-lens 576 --kernel-size 11 --lr 5e-4 --weight-decay 0 --output-dir "${RUNS}/cfg_576_k11_lr5e4"
python -u train_dlinear_sweep.py ${COMMON_TRAIN} ${COMMON_LOSS} --no-window-plots --input-lens 576 --kernel-size 17 --lr 5e-4 --weight-decay 0 --output-dir "${RUNS}/cfg_576_k17_lr5e4"
python -u train_dlinear_sweep.py ${COMMON_TRAIN} ${COMMON_LOSS} --no-window-plots --input-lens 432 --kernel-size 11 --lr 2e-4 --weight-decay 1e-6 --output-dir "${RUNS}/cfg_432_k11_lr2e4"
python -u train_dlinear_sweep.py ${COMMON_TRAIN} ${COMMON_LOSS} --no-window-plots --input-lens 864 --kernel-size 17 --lr 2e-4 --weight-decay 1e-6 --output-dir "${RUNS}/cfg_864_k17_lr2e4"

# 2) Rank all candidate runs and store best config summary.
python -u select_best_dlinear_run.py --runs-root "${RUNS}" --out-json "best_run_selection.json"

# 3) Re-run only the selected best candidate with window plots enabled.
python -u rerun_best_dlinear_with_plots.py \
	--selection-json "${RUNS}/best_run_selection.json" \
	--output-root "${BEST}" \
	${COMMON_TRAIN} \
	${COMMON_LOSS}

python -u select_best_dlinear_run.py --runs-root "${BEST}" --out-json "best_model_selection.json"

echo "DLinear tuning completed."
echo "Candidates ranking: ${RUNS}/dlinear_run_ranking.csv"
echo "Selected best candidate metadata: ${RUNS}/best_run_selection.json"
echo "Best-model-only window plots under: ${BEST}/<best_cfg>/rolling_window_forecasts/plots"
