#!/bin/bash
#SBATCH -A prj0000000262
#SBATCH -p testqueue
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 10:00:00
#SBATCH -J transformer_tuning
#SBATCH -o ~/logs/%x_%j.out
#SBATCH -e ~/logs/%x_%j.err

set -euo pipefail

mkdir -p ~/logs

# Uniform-step window filter (5-min cadence by default; override via env, e.g. UNIFORM_STEP_SECONDS=1800).
UNIFORM_STEP_SECONDS=${UNIFORM_STEP_SECONDS:-300}
UNIFORM_TOL_SECONDS=${UNIFORM_TOL_SECONDS:-30}
# No post-smoothing on forecast horizons (--pred-smoothing-window 1).
COMMON_TRAIN="--epochs 250 --batch-size 16 --early-stopping-patience 20 --scheduler-patience 6 --pred-lens 288 --pred-smoothing-window 1 --uniform-step-seconds ${UNIFORM_STEP_SECONDS} --uniform-step-tolerance-seconds ${UNIFORM_TOL_SECONDS}"
# Tail loss weight: 1.0 = flat (default; prevents tail spikes). Pass LOSS_TAIL_WEIGHT=0.8 to restore the old decay.
LOSS_TAIL_WEIGHT=${LOSS_TAIL_WEIGHT:-1.0}
COMMON_LOSS="--loss-point-weight 0.7 --loss-diff-weight 0.9 --loss-curvature-weight 0.5 --loss-variance-weight 0.2 --loss-tail-weight ${LOSS_TAIL_WEIGHT}"
TARGET_COLUMN=${TARGET_COLUMN:-"Acceleration RMS"}
ROLLING_WINDOW_ARTIFACT_LIMIT=${ROLLING_WINDOW_ARTIFACT_LIMIT:-20}
COMMON_ROLLING="--rolling-window-artifact-limit ${ROLLING_WINDOW_ARTIFACT_LIMIT}"

# Input feature selection. Two ways to override at submit time:
#   1) INPUT_FEATURES — semicolon-separated column names (so column names with spaces work).
#        sbatch --export=ALL,INPUT_FEATURES="Acceleration RMS;Kurtosis;Battery Voltage" run_transformer_tuning.sbatch
#   2) USE_ALL_NUMERIC_FEATURES=1 (default) uses every numeric column except TIMESTAMP / smoothed target.
#        Set USE_ALL_NUMERIC_FEATURES=0 (and leave INPUT_FEATURES empty) to fall back to TARGET_COLUMN only.
INPUT_FEATURES=${INPUT_FEATURES:-}
USE_ALL_NUMERIC_FEATURES=${USE_ALL_NUMERIC_FEATURES:-0}
if [ -n "${INPUT_FEATURES}" ]; then
  IFS=';' read -r -a INPUT_FEATURE_ARR <<< "${INPUT_FEATURES}"
  FEATURE_FLAGS=(--feature-columns "${INPUT_FEATURE_ARR[@]}")
  echo "Input features (explicit): ${INPUT_FEATURE_ARR[@]}"
elif [ "${USE_ALL_NUMERIC_FEATURES}" = "1" ]; then
  FEATURE_FLAGS=(--use-all-numeric-features)
  echo "Input features: all numeric columns (default)."
else
  FEATURE_FLAGS=()
  echo "Input features: target column only (${TARGET_COLUMN})."
fi

TRAIN_VAL_CSV=${TRAIN_VAL_CSV:-}
TEST_CSV=${TEST_CSV:-}
TRAIN_CSV=${TRAIN_CSV:-}
VAL_CSV=${VAL_CSV:-}
SINGLE_CSV=${SINGLE_CSV:-data/AHU_2_9_Blower_DE_A.csv}
TRAIN_RATIO=${TRAIN_RATIO:-0.80}
VAL_RATIO=${VAL_RATIO:-0.10}

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

MODEL_SLUG="transformer"
TRAIN_SCRIPT="train_transformer_sweep.py"

ROOT="outputs_${MODEL_SLUG}_tuning${DATA_FOLDER_SUFFIX}_${DATA_TAG}"
RUNS="${ROOT}/runs"
BEST="${ROOT}/best_model"
mkdir -p "${RUNS}" "${BEST}"

python -u ${TRAIN_SCRIPT} ${COMMON_TRAIN} ${COMMON_LOSS} --value-column "${TARGET_COLUMN}" "${FEATURE_FLAGS[@]}" ${DATA_FLAGS} --save-window-plots ${COMMON_ROLLING} --input-lens 864 --d-model 128 --nhead 8 --num-layers 4 --dim-feedforward 256 --lr 5e-4 --weight-decay 0 --output-dir "${RUNS}/cfg_864_d128_h8_l4_lr5e4"
python -u ${TRAIN_SCRIPT} ${COMMON_TRAIN} ${COMMON_LOSS} --value-column "${TARGET_COLUMN}" "${FEATURE_FLAGS[@]}" ${DATA_FLAGS} --save-window-plots ${COMMON_ROLLING} --input-lens 1296 --d-model 128 --nhead 8 --num-layers 5 --dim-feedforward 256 --lr 5e-4 --weight-decay 0 --output-dir "${RUNS}/cfg_1296_d128_h8_l5_lr5e4"
python -u ${TRAIN_SCRIPT} ${COMMON_TRAIN} ${COMMON_LOSS} --value-column "${TARGET_COLUMN}" "${FEATURE_FLAGS[@]}" ${DATA_FLAGS} --save-window-plots ${COMMON_ROLLING} --input-lens 1728 --d-model 160 --nhead 8 --num-layers 5 --dim-feedforward 320 --lr 2e-4 --weight-decay 1e-6 --output-dir "${RUNS}/cfg_1728_d160_h8_l5_lr2e4"
python -u ${TRAIN_SCRIPT} ${COMMON_TRAIN} ${COMMON_LOSS} --value-column "${TARGET_COLUMN}" "${FEATURE_FLAGS[@]}" ${DATA_FLAGS} --save-window-plots ${COMMON_ROLLING} --input-lens 2160 --d-model 192 --nhead 8 --num-layers 6 --dim-feedforward 384 --lr 2e-4 --weight-decay 1e-6 --output-dir "${RUNS}/cfg_2160_d192_h8_l6_lr2e4"
python -u ${TRAIN_SCRIPT} ${COMMON_TRAIN} ${COMMON_LOSS} --value-column "${TARGET_COLUMN}" "${FEATURE_FLAGS[@]}" ${DATA_FLAGS} --save-window-plots ${COMMON_ROLLING} --input-lens 2592 --d-model 192 --nhead 8 --num-layers 6 --dim-feedforward 384 --lr 2e-4 --weight-decay 1e-6 --output-dir "${RUNS}/cfg_2592_d192_h8_l6_lr2e4"

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
  --value-column "${TARGET_COLUMN}" \
  "${FEATURE_FLAGS[@]}" \
  ${COMMON_ROLLING} \
  ${COMMON_TRAIN} \
  ${COMMON_LOSS}

python -u select_best_sweep_run.py --runs-root "${BEST}" --out-json "best_model_selection.json"

echo "Transformer tuning completed."
echo "Candidates ranking: ${RUNS}/sweep_run_ranking.csv"
echo "Selected best candidate metadata: ${RUNS}/best_run_selection.json"
echo "Candidate run plots under: ${RUNS}/<cfg>/rolling_window_forecasts/plots"
echo "Best-model-only window plots under: ${BEST}/<best_cfg>/rolling_window_forecasts/plots"
