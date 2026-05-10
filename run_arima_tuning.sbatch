#!/bin/bash
#SBATCH -A prj0000000262
#SBATCH -p testqueue
#SBATCH -N 1
#SBATCH --gpus=0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 10:00:00
#SBATCH -J arima_tuning
#SBATCH -o ~/logs/%x_%j.out
#SBATCH -e ~/logs/%x_%j.err

set -euo pipefail

mkdir -p ~/logs

# Uniform-step window filter (same defaults as run_dlinear_tuning.sbatch).
UNIFORM_STEP_SECONDS=${UNIFORM_STEP_SECONDS:-300}
UNIFORM_TOL_SECONDS=${UNIFORM_TOL_SECONDS:-30}
# Post-smoothing on forecast horizons only (--pred-smoothing-window 1 = off).
COMMON_WIN="--pred-lens 288 --pred-smoothing-window 1 --uniform-step-seconds ${UNIFORM_STEP_SECONDS} --uniform-step-tolerance-seconds ${UNIFORM_TOL_SECONDS}"
ROLLING_WINDOW_ARTIFACT_LIMIT=${ROLLING_WINDOW_ARTIFACT_LIMIT:-20}
COMMON_ROLLING="--rolling-window-artifact-limit ${ROLLING_WINDOW_ARTIFACT_LIMIT}"

# --- Target smoothing BEFORE train/val/test splits (per CSV file) ---
# Centered moving average length K on --value-column. K=1 disables.
# Example: sbatch --export=ALL,TARGET_SMOOTHING_WINDOW=7 run_arima_tuning.sbatch
TARGET_SMOOTHING_WINDOW=${TARGET_SMOOTHING_WINDOW:-1}
TARGET_SMOOTH_FLAGS=()
if [ "${TARGET_SMOOTHING_WINDOW}" -gt 1 ] 2>/dev/null; then
  TARGET_SMOOTH_FLAGS=(--target-smoothing-window "${TARGET_SMOOTHING_WINDOW}")
  echo "Pre-split target smoothing: window=${TARGET_SMOOTHING_WINDOW} (centered MA on value column per CSV)."
else
  echo "Pre-split target smoothing: off (TARGET_SMOOTHING_WINDOW=${TARGET_SMOOTHING_WINDOW})."
fi

TARGET_COLUMN=${TARGET_COLUMN:-"Acceleration RMS"}

# statsmodels is required (ARIMA). Install in your env if needed: pip install statsmodels
ARIMA_P=${ARIMA_P:-2}
ARIMA_D=${ARIMA_D:-1}
ARIMA_Q=${ARIMA_Q:-2}
ARIMA_MAXITER=${ARIMA_MAXITER:-100}
COMMON_ARIMA_ORDER=(--arima-p "${ARIMA_P}" --arima-d "${ARIMA_D}" --arima-q "${ARIMA_Q}" --arima-maxiter "${ARIMA_MAXITER}")

# Input feature flags are accepted for parity with DLinear jobs but ignored by train_arima_sweep.py (univariate).
INPUT_FEATURES=${INPUT_FEATURES:-}
USE_ALL_NUMERIC_FEATURES=${USE_ALL_NUMERIC_FEATURES:-0}
if [ -n "${INPUT_FEATURES}" ]; then
  IFS=';' read -r -a INPUT_FEATURE_ARR <<< "${INPUT_FEATURES}"
  FEATURE_FLAGS=(--feature-columns "${INPUT_FEATURE_ARR[@]}")
  echo "Input features (explicit, ignored by ARIMA): ${INPUT_FEATURE_ARR[*]}"
elif [ "${USE_ALL_NUMERIC_FEATURES}" = "1" ]; then
  FEATURE_FLAGS=(--use-all-numeric-features)
  echo "Input features: USE_ALL_NUMERIC_FEATURES=1 (ignored by ARIMA)."
else
  FEATURE_FLAGS=()
fi

# Dataset selection — same three modes as run_dlinear_tuning.sbatch.
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

# Under sbatch, SLURM_JOB_ID is the scheduler job id — append so outputs never collide across submissions.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  JOB_SUFFIX="_job${SLURM_JOB_ID}"
else
  JOB_SUFFIX=""
fi

TRAIN_SCRIPT="train_arima_sweep.py"
RUN_TAG="arima_p${ARIMA_P}_d${ARIMA_D}_q${ARIMA_Q}"

ROOT="outputs_arima_tuning_${RUN_TAG}${DATA_FOLDER_SUFFIX}_${DATA_TAG}${JOB_SUFFIX}"
RUNS="${ROOT}/runs"
BEST="${ROOT}/best_model"
mkdir -p "${RUNS}" "${BEST}"
echo "Output directory root: ${ROOT}"

# Dummy training args — required by add_common_args / rerun_best_sweep_with_plots parity (ARIMA ignores epochs).
STUB_TRAIN="--epochs 1 --batch-size 8 --early-stopping-patience 1 --scheduler-patience 1"

# Candidate grid: same input lengths as run_dlinear_tuning.sbatch (history length for each rolling fit).
python -u ${TRAIN_SCRIPT} ${STUB_TRAIN} ${COMMON_WIN} "${TARGET_SMOOTH_FLAGS[@]}" "${COMMON_ARIMA_ORDER[@]}" --value-column "${TARGET_COLUMN}" "${FEATURE_FLAGS[@]}" ${DATA_FLAGS} --save-window-plots ${COMMON_ROLLING} --input-lens 864 --output-dir "${RUNS}/cfg_864_${RUN_TAG}"
python -u ${TRAIN_SCRIPT} ${STUB_TRAIN} ${COMMON_WIN} "${TARGET_SMOOTH_FLAGS[@]}" "${COMMON_ARIMA_ORDER[@]}" --value-column "${TARGET_COLUMN}" "${FEATURE_FLAGS[@]}" ${DATA_FLAGS} --save-window-plots ${COMMON_ROLLING} --input-lens 1296 --output-dir "${RUNS}/cfg_1296_${RUN_TAG}"
python -u ${TRAIN_SCRIPT} ${STUB_TRAIN} ${COMMON_WIN} "${TARGET_SMOOTH_FLAGS[@]}" "${COMMON_ARIMA_ORDER[@]}" --value-column "${TARGET_COLUMN}" "${FEATURE_FLAGS[@]}" ${DATA_FLAGS} --save-window-plots ${COMMON_ROLLING} --input-lens 1728 --output-dir "${RUNS}/cfg_1728_${RUN_TAG}"
python -u ${TRAIN_SCRIPT} ${STUB_TRAIN} ${COMMON_WIN} "${TARGET_SMOOTH_FLAGS[@]}" "${COMMON_ARIMA_ORDER[@]}" --value-column "${TARGET_COLUMN}" "${FEATURE_FLAGS[@]}" ${DATA_FLAGS} --save-window-plots ${COMMON_ROLLING} --input-lens 2160 --output-dir "${RUNS}/cfg_2160_${RUN_TAG}"
python -u ${TRAIN_SCRIPT} ${STUB_TRAIN} ${COMMON_WIN} "${TARGET_SMOOTH_FLAGS[@]}" "${COMMON_ARIMA_ORDER[@]}" --value-column "${TARGET_COLUMN}" "${FEATURE_FLAGS[@]}" ${DATA_FLAGS} --save-window-plots ${COMMON_ROLLING} --input-lens 2592 --output-dir "${RUNS}/cfg_2592_${RUN_TAG}"

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
  ${STUB_TRAIN} \
  --pred-lens 288 \
  --pred-smoothing-window 1 \
  --target-smoothing-window "${TARGET_SMOOTHING_WINDOW}"

python -u select_best_sweep_run.py --runs-root "${BEST}" --out-json "best_model_selection.json"

echo "ARIMA tuning completed. Order=(${ARIMA_P},${ARIMA_D},${ARIMA_Q}) tag=${RUN_TAG}"
echo "Candidates ranking: ${RUNS}/sweep_run_ranking.csv"
echo "Best rerun metadata: ${BEST}/<best_cfg>/best_config.json"
echo "Plots under: ${RUNS}/<cfg>/rolling_window_forecasts/plots"
