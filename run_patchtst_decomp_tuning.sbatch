#!/bin/bash
#SBATCH -A prj0000000262
#SBATCH -p h200p
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -J patchtst_decomp_tuning
#SBATCH -o ~/logs/%x_%j.out
#SBATCH -e ~/logs/%x_%j.err

set -euo pipefail

mkdir -p ~/logs

# Uniform-step window filter (5-min cadence by default; override via env, e.g. UNIFORM_STEP_SECONDS=1800).
UNIFORM_STEP_SECONDS=${UNIFORM_STEP_SECONDS:-300}
UNIFORM_TOL_SECONDS=${UNIFORM_TOL_SECONDS:-30}
# No post-smoothing on forecast horizons (--pred-smoothing-window 1).
COMMON_TRAIN="--epochs 250 --batch-size 16 --early-stopping-patience 20 --scheduler-patience 6 --pred-lens 288 --pred-smoothing-window 1 --uniform-step-seconds ${UNIFORM_STEP_SECONDS} --uniform-step-tolerance-seconds ${UNIFORM_TOL_SECONDS}"
COMMON_LOSS="--loss-point-weight 0.7 --loss-diff-weight 0.9 --loss-curvature-weight 0.5 --loss-variance-weight 0.2"
ROLLING_WINDOW_ARTIFACT_LIMIT=${ROLLING_WINDOW_ARTIFACT_LIMIT:-20}
COMMON_ROLLING="--rolling-window-artifact-limit ${ROLLING_WINDOW_ARTIFACT_LIMIT}"

TRAIN_VAL_CSV=${TRAIN_VAL_CSV:-data_train_val.csv}
TEST_CSV=${TEST_CSV:-data_test_anomalous.csv}
TRAIN_CSV=${TRAIN_CSV:-}
VAL_CSV=${VAL_CSV:-}
SINGLE_CSV=${SINGLE_CSV:-}
TRAIN_RATIO=${TRAIN_RATIO:-0.70}
VAL_RATIO=${VAL_RATIO:-0.15}

if [ -n "${SINGLE_CSV}" ]; then
  DATA_FLAGS="--single-csv ${SINGLE_CSV} --train-ratio ${TRAIN_RATIO} --val-ratio ${VAL_RATIO}"
  if [ -n "${TEST_RATIO:-}" ]; then
    DATA_FLAGS="${DATA_FLAGS} --test-ratio ${TEST_RATIO}"
  fi
  DATA_TAG="$(basename "${SINGLE_CSV}" .csv)_single_r${TRAIN_RATIO}_v${VAL_RATIO}"
  REF_PARENT="$(dirname "${SINGLE_CSV}")"
elif [ -n "${TRAIN_CSV}" ] && [ -n "${VAL_CSV}" ]; then
  DATA_FLAGS="--train-csv ${TRAIN_CSV} --val-csv ${VAL_CSV} --test-csv ${TEST_CSV}"
  DATA_TAG="$(basename "${TRAIN_CSV}" .csv)__$(basename "${VAL_CSV}" .csv)__to__$(basename "${TEST_CSV}" .csv)"
  REF_PARENT="$(dirname "${TRAIN_CSV}")"
else
  DATA_FLAGS="--train-val-csv ${TRAIN_VAL_CSV} --test-csv ${TEST_CSV}"
  TRAIN_VAL_BASENAME="$(basename "${TRAIN_VAL_CSV}" .csv)"
  TEST_BASENAME="$(basename "${TEST_CSV}" .csv)"
  DATA_TAG="${TRAIN_VAL_BASENAME}_to_${TEST_BASENAME}"
  REF_PARENT="$(dirname "${TRAIN_VAL_CSV}")"
fi
DATA_FOLDER_NAME="$(basename "${REF_PARENT}")"
if [ "${DATA_FOLDER_NAME}" = "." ]; then
  DATA_FOLDER_SUFFIX=""
else
  DATA_FOLDER_SUFFIX="_${DATA_FOLDER_NAME}"
fi

MODEL_SLUG="patchtst_decomp"
TRAIN_SCRIPT="train_patchtst_decomp_sweep.py"

ROOT="outputs_${MODEL_SLUG}_tuning${DATA_FOLDER_SUFFIX}_${DATA_TAG}"
RUNS="${ROOT}/runs"
BEST="${ROOT}/best_model"
mkdir -p "${RUNS}" "${BEST}"

python -u ${TRAIN_SCRIPT} ${COMMON_TRAIN} ${COMMON_LOSS} ${DATA_FLAGS} --save-window-plots ${COMMON_ROLLING} --input-lens 864 --patch-len 16 --patch-stride 8 --d-model 128 --num-layers 3 --lr 5e-4 --weight-decay 0 --output-dir "${RUNS}/cfg_864_p16_s8_d128_l3_lr5e4"
python -u ${TRAIN_SCRIPT} ${COMMON_TRAIN} ${COMMON_LOSS} ${DATA_FLAGS} --save-window-plots ${COMMON_ROLLING} --input-lens 1296 --patch-len 24 --patch-stride 12 --d-model 128 --num-layers 3 --lr 5e-4 --weight-decay 0 --output-dir "${RUNS}/cfg_1296_p24_s12_d128_l3_lr5e4"
python -u ${TRAIN_SCRIPT} ${COMMON_TRAIN} ${COMMON_LOSS} ${DATA_FLAGS} --save-window-plots ${COMMON_ROLLING} --input-lens 1728 --patch-len 24 --patch-stride 12 --d-model 160 --num-layers 4 --lr 2e-4 --weight-decay 1e-6 --output-dir "${RUNS}/cfg_1728_p24_s12_d160_l4_lr2e4"
python -u ${TRAIN_SCRIPT} ${COMMON_TRAIN} ${COMMON_LOSS} ${DATA_FLAGS} --save-window-plots ${COMMON_ROLLING} --input-lens 2160 --patch-len 32 --patch-stride 16 --d-model 192 --num-layers 4 --lr 2e-4 --weight-decay 1e-6 --output-dir "${RUNS}/cfg_2160_p32_s16_d192_l4_lr2e4"
python -u ${TRAIN_SCRIPT} ${COMMON_TRAIN} ${COMMON_LOSS} ${DATA_FLAGS} --save-window-plots ${COMMON_ROLLING} --input-lens 2592 --patch-len 32 --patch-stride 16 --d-model 192 --num-layers 4 --lr 2e-4 --weight-decay 1e-6 --output-dir "${RUNS}/cfg_2592_p32_s16_d192_l4_lr2e4"

python -u select_best_sweep_run.py --runs-root "${RUNS}" --out-json "best_run_selection.json"

if [ -n "${SINGLE_CSV}" ]; then
  RERUN_DATA=(--single-csv "${SINGLE_CSV}" --train-ratio "${TRAIN_RATIO}" --val-ratio "${VAL_RATIO}")
  if [ -n "${TEST_RATIO:-}" ]; then
    RERUN_DATA+=(--test-ratio "${TEST_RATIO}")
  fi
elif [ -n "${TRAIN_CSV}" ] && [ -n "${VAL_CSV}" ]; then
  RERUN_DATA=(--train-csv "${TRAIN_CSV}" --val-csv "${VAL_CSV}" --test-csv "${TEST_CSV}")
else
  RERUN_DATA=(--train-val-csv "${TRAIN_VAL_CSV}" --test-csv "${TEST_CSV}")
fi
python -u rerun_best_sweep_with_plots.py \
  --selection-json "${RUNS}/best_run_selection.json" \
  --train-script "${TRAIN_SCRIPT}" \
  --output-root "${BEST}" \
  "${RERUN_DATA[@]}" \
  ${COMMON_ROLLING} \
  ${COMMON_TRAIN} \
  ${COMMON_LOSS}

python -u select_best_sweep_run.py --runs-root "${BEST}" --out-json "best_model_selection.json"

echo "PatchTST+Decomposition tuning completed."
echo "Candidates ranking: ${RUNS}/sweep_run_ranking.csv"
echo "Selected best candidate metadata: ${RUNS}/best_run_selection.json"
echo "Candidate run plots under: ${RUNS}/<cfg>/rolling_window_forecasts/plots"
echo "Best-model-only window plots under: ${BEST}/<best_cfg>/rolling_window_forecasts/plots"
