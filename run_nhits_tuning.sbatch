#!/bin/bash
#SBATCH -A prj0000000262
#SBATCH -p h200p
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -J nhits_tuning
#SBATCH -o ~/logs/%x_%j.out
#SBATCH -e ~/logs/%x_%j.err

set -euo pipefail

mkdir -p ~/logs

ROOT=outputs_nhits_tuning
RUNS=${ROOT}/runs
BEST=${ROOT}/best_model
mkdir -p "${RUNS}" "${BEST}"

COMMON_TRAIN="--epochs 250 --batch-size 16 --early-stopping-patience 20 --scheduler-patience 6 --pred-lens 288"
COMMON_LOSS="--loss-point-weight 0.7 --loss-diff-weight 0.9 --loss-curvature-weight 0.5 --loss-variance-weight 0.2"

python -u train_nhits_sweep.py ${COMMON_TRAIN} ${COMMON_LOSS} --save-window-plots --input-lens 864 --hidden-dim 256 --n-blocks 3 --lr 5e-4 --weight-decay 0 --output-dir "${RUNS}/cfg_864_h256_b3_lr5e4"
python -u train_nhits_sweep.py ${COMMON_TRAIN} ${COMMON_LOSS} --save-window-plots --input-lens 1296 --hidden-dim 256 --n-blocks 4 --lr 5e-4 --weight-decay 0 --output-dir "${RUNS}/cfg_1296_h256_b4_lr5e4"
python -u train_nhits_sweep.py ${COMMON_TRAIN} ${COMMON_LOSS} --save-window-plots --input-lens 1728 --hidden-dim 320 --n-blocks 4 --lr 2e-4 --weight-decay 1e-6 --output-dir "${RUNS}/cfg_1728_h320_b4_lr2e4"
python -u train_nhits_sweep.py ${COMMON_TRAIN} ${COMMON_LOSS} --save-window-plots --input-lens 2160 --hidden-dim 384 --n-blocks 4 --lr 2e-4 --weight-decay 1e-6 --output-dir "${RUNS}/cfg_2160_h384_b4_lr2e4"

python -u select_best_sweep_run.py --runs-root "${RUNS}" --out-json "best_run_selection.json"

python -u rerun_best_sweep_with_plots.py \
  --selection-json "${RUNS}/best_run_selection.json" \
  --train-script "train_nhits_sweep.py" \
  --output-root "${BEST}" \
  ${COMMON_TRAIN} \
  ${COMMON_LOSS}

python -u select_best_sweep_run.py --runs-root "${BEST}" --out-json "best_model_selection.json"

echo "N-HiTS tuning completed."
echo "Candidates ranking: ${RUNS}/sweep_run_ranking.csv"
echo "Selected best candidate metadata: ${RUNS}/best_run_selection.json"
echo "Candidate run plots under: ${RUNS}/<cfg>/rolling_window_forecasts/plots"
echo "Best-model-only window plots under: ${BEST}/<best_cfg>/rolling_window_forecasts/plots"
