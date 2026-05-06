#!/bin/bash
#SBATCH -A prj0000000262
#SBATCH -p h200p
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -J transformer_tuning
#SBATCH -o ~/logs/%x_%j.out
#SBATCH -e ~/logs/%x_%j.err

set -euo pipefail

mkdir -p ~/logs

COMMON_TRAIN="--epochs 250 --batch-size 16 --early-stopping-patience 20 --scheduler-patience 6 --pred-lens 288"
COMMON_LOSS="--loss-point-weight 0.7 --loss-diff-weight 0.9 --loss-curvature-weight 0.5 --loss-variance-weight 0.2"
ROLLING_WINDOW_ARTIFACT_LIMIT=${ROLLING_WINDOW_ARTIFACT_LIMIT:-20}
COMMON_ROLLING="--rolling-window-artifact-limit ${ROLLING_WINDOW_ARTIFACT_LIMIT}"

TRAIN_VAL_CSV=${TRAIN_VAL_CSV:-data_train_val.csv}
TEST_CSV=${TEST_CSV:-data_test_anomalous.csv}
TRAIN_VAL_BASENAME="$(basename "${TRAIN_VAL_CSV}" .csv)"
TEST_BASENAME="$(basename "${TEST_CSV}" .csv)"
DATA_TAG="${TRAIN_VAL_BASENAME}_to_${TEST_BASENAME}"
TRAIN_VAL_PARENT="$(dirname "${TRAIN_VAL_CSV}")"
DATA_FOLDER_NAME="$(basename "${TRAIN_VAL_PARENT}")"
if [ "${DATA_FOLDER_NAME}" = "." ]; then
  DATA_FOLDER_SUFFIX=""
else
  DATA_FOLDER_SUFFIX="_${DATA_FOLDER_NAME}"
fi

MODEL_SLUG="transformer"
TRAIN_SCRIPT="train_transformer_sweep.py"
DATA_FLAGS="--train-val-csv ${TRAIN_VAL_CSV} --test-csv ${TEST_CSV}"

ROOT="outputs_${MODEL_SLUG}_tuning${DATA_FOLDER_SUFFIX}_${DATA_TAG}"
RUNS="${ROOT}/runs"
BEST="${ROOT}/best_model"
mkdir -p "${RUNS}" "${BEST}"

python -u ${TRAIN_SCRIPT} ${COMMON_TRAIN} ${COMMON_LOSS} ${DATA_FLAGS} --save-window-plots ${COMMON_ROLLING} --input-lens 864 --d-model 128 --nhead 8 --num-layers 4 --dim-feedforward 256 --lr 5e-4 --weight-decay 0 --output-dir "${RUNS}/cfg_864_d128_h8_l4_lr5e4"
python -u ${TRAIN_SCRIPT} ${COMMON_TRAIN} ${COMMON_LOSS} ${DATA_FLAGS} --save-window-plots ${COMMON_ROLLING} --input-lens 1296 --d-model 128 --nhead 8 --num-layers 5 --dim-feedforward 256 --lr 5e-4 --weight-decay 0 --output-dir "${RUNS}/cfg_1296_d128_h8_l5_lr5e4"
python -u ${TRAIN_SCRIPT} ${COMMON_TRAIN} ${COMMON_LOSS} ${DATA_FLAGS} --save-window-plots ${COMMON_ROLLING} --input-lens 1728 --d-model 160 --nhead 8 --num-layers 5 --dim-feedforward 320 --lr 2e-4 --weight-decay 1e-6 --output-dir "${RUNS}/cfg_1728_d160_h8_l5_lr2e4"
python -u ${TRAIN_SCRIPT} ${COMMON_TRAIN} ${COMMON_LOSS} ${DATA_FLAGS} --save-window-plots ${COMMON_ROLLING} --input-lens 2160 --d-model 192 --nhead 8 --num-layers 6 --dim-feedforward 384 --lr 2e-4 --weight-decay 1e-6 --output-dir "${RUNS}/cfg_2160_d192_h8_l6_lr2e4"
python -u ${TRAIN_SCRIPT} ${COMMON_TRAIN} ${COMMON_LOSS} ${DATA_FLAGS} --save-window-plots ${COMMON_ROLLING} --input-lens 2592 --d-model 192 --nhead 8 --num-layers 6 --dim-feedforward 384 --lr 2e-4 --weight-decay 1e-6 --output-dir "${RUNS}/cfg_2592_d192_h8_l6_lr2e4"

python -u select_best_sweep_run.py --runs-root "${RUNS}" --out-json "best_run_selection.json"

python -u rerun_best_sweep_with_plots.py \
  --selection-json "${RUNS}/best_run_selection.json" \
  --train-script "${TRAIN_SCRIPT}" \
  --output-root "${BEST}" \
  --train-val-csv "${TRAIN_VAL_CSV}" \
  --test-csv "${TEST_CSV}" \
  ${COMMON_ROLLING} \
  ${COMMON_TRAIN} \
  ${COMMON_LOSS}

python -u select_best_sweep_run.py --runs-root "${BEST}" --out-json "best_model_selection.json"

echo "Transformer tuning completed."
echo "Candidates ranking: ${RUNS}/sweep_run_ranking.csv"
echo "Selected best candidate metadata: ${RUNS}/best_run_selection.json"
echo "Candidate run plots under: ${RUNS}/<cfg>/rolling_window_forecasts/plots"
echo "Best-model-only window plots under: ${BEST}/<best_cfg>/rolling_window_forecasts/plots"
