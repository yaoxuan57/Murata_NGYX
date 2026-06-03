#!/bin/bash
#SBATCH -A prj0000000262
#SBATCH -p testqueue
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 10:00:00
#SBATCH -J dlinear_tuning
#SBATCH -o ~/logs/%x_%j.out
#SBATCH -e ~/logs/%x_%j.err

set -euo pipefail

mkdir -p ~/logs

# --- Timestamp / window rules (see forecast_sweep_common.compute_timestep_window_start_indices) ---
# Max gap: every consecutive row-to-row Δt inside a window must be ≤ this (seconds). Any shorter cadence is OK.
MAX_CONSECUTIVE_TIMESTAMP_GAP_SECONDS=${MAX_CONSECUTIVE_TIMESTAMP_GAP_SECONDS:-1860}
# Nominal cadence (optional): set REQUIRE_UNIFORM_TIMESTEP=1 to also require each step ≈ UNIFORM_STEP_SECONDS ± UNIFORM_TOL_SECONDS.
# Default 0 = do NOT use nominal spacing; only the max-gap rule above filters windows.
REQUIRE_UNIFORM_TIMESTEP=${REQUIRE_UNIFORM_TIMESTEP:-1}
UNIFORM_STEP_SECONDS=${UNIFORM_STEP_SECONDS:-1800}
UNIFORM_TOL_SECONDS=${UNIFORM_TOL_SECONDS:-60}
if [ "${REQUIRE_UNIFORM_TIMESTEP}" = "1" ]; then
  _UNIFORM_FLAGS="--require-uniform-timestep --uniform-step-seconds ${UNIFORM_STEP_SECONDS} --uniform-step-tolerance-seconds ${UNIFORM_TOL_SECONDS}"
else
  _UNIFORM_FLAGS="--no-require-uniform-timestep"
fi
# No post-smoothing on forecast horizons (--pred-smoothing-window 1).
COMMON_TRAIN="--epochs 500 --batch-size 16 --early-stopping-patience 50 --min-delta 1e-6 --scheduler-patience 6 --pred-lens 48 --pred-smoothing-window 1 ${_UNIFORM_FLAGS} --max-consecutive-timestamp-gap-seconds ${MAX_CONSECUTIVE_TIMESTAMP_GAP_SECONDS} --train-window-stride 1 --val-window-stride 0 --test-window-stride 0"
# Tail loss weight: 1.0 = flat (default; prevents tail spikes). Pass LOSS_TAIL_WEIGHT=0.8 to restore the old decay.
LOSS_TAIL_WEIGHT=${LOSS_TAIL_WEIGHT:-1.0}
COMMON_LOSS="--loss-point-weight 0.7 --loss-diff-weight 0.9 --loss-curvature-weight 0.5 --loss-variance-weight 0.2 --loss-tail-weight ${LOSS_TAIL_WEIGHT}"
# AdamW decoupled weight decay (PyTorch --weight-decay). Default 1e-4 is a common starting point for small transformers;
# use 0 to match older runs, or try 1e-3 / 1e-2 if you still see overfit (often pair smaller --lr with larger decay).
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
echo "AdamW weight decay: ${WEIGHT_DECAY} (override: sbatch --export=ALL,WEIGHT_DECAY=1e-3,...)"
# Per-step prediction band (pinball / quantile loss). Saved in best_config model_config for rerun.
# Disable: sbatch --export=ALL,ENABLE_FORECAST_QUANTILES=0 run_dlinear_tuning.sbatch
ENABLE_FORECAST_QUANTILES=${ENABLE_FORECAST_QUANTILES:-1}
COMMON_QUANT=()
if [ "${ENABLE_FORECAST_QUANTILES}" != "0" ] && [ "${ENABLE_FORECAST_QUANTILES}" != "no" ]; then
  COMMON_QUANT=(--forecast-quantiles 0.05 0.5 0.95)
  echo "Quantile forecast: ${COMMON_QUANT[*]}"
else
  echo "Quantile forecast: off (point-only; ENABLE_FORECAST_QUANTILES=${ENABLE_FORECAST_QUANTILES})."
fi
TARGET_COLUMN=${TARGET_COLUMN:-"Acceleration RMS"}
# Per-window rolling CSV/PNG under rolling_window_forecasts/ (full series still in all_windows_forecasts.csv).
# Default: write artifacts for every test window. To cap disk use, set e.g.
#   sbatch --export=ALL,ROLLING_WINDOW_ARTIFACT_LIMIT=30 run_dlinear_tuning.sbatch
ROLLING_WINDOW_ARTIFACT_LIMIT=${ROLLING_WINDOW_ARTIFACT_LIMIT:-}
COMMON_ROLLING=()
if [ -n "${ROLLING_WINDOW_ARTIFACT_LIMIT}" ] && [ "${ROLLING_WINDOW_ARTIFACT_LIMIT}" -gt 0 ] 2>/dev/null; then
  COMMON_ROLLING=(--rolling-window-artifact-limit "${ROLLING_WINDOW_ARTIFACT_LIMIT}")
  echo "Rolling-window artifact limit: ${ROLLING_WINDOW_ARTIFACT_LIMIT} (evenly spaced PNG+CSV)."
else
  echo "Rolling-window artifacts: all windows (PNG+CSV per window)."
fi

# Stitched full-test interactive HTML (Plotly) under rolling_window_forecasts/stitched_test_forecast.html:
# first window's actual input, then all test horizons on real timestamps (median, band, actual).
# Requires plotly on the compute node:  pip install plotly
# Disable: sbatch --export=ALL,SAVE_STITCHED_TEST_HTML=0 run_dlinear_tuning.sbatch
SAVE_STITCHED_TEST_HTML=${SAVE_STITCHED_TEST_HTML:-1}
STITCH_HTML_FLAGS=()
if [ "${SAVE_STITCHED_TEST_HTML}" != "0" ] && [ "${SAVE_STITCHED_TEST_HTML}" != "no" ]; then
  STITCH_HTML_FLAGS=(--save-stitched-test-html)
  echo "Stitched test HTML: on (SAVE_STITCHED_TEST_HTML=${SAVE_STITCHED_TEST_HTML})."
else
  echo "Stitched test HTML: off."
fi

# Pre-split smoothing on --value-column: causal (trailing) MA per CSV before windows (window length = TARGET_SMOOTHING_WINDOW).
# Rolling-window PNGs plot that same smoothed value history and smoothed horizons (actual); predictions are raw model outputs unless --pred-smoothing-window>1 is set in train args.
# Example: sbatch --export=ALL,TARGET_SMOOTHING_WINDOW=7 run_dlinear_tuning.sbatch
TARGET_SMOOTHING_WINDOW=${TARGET_SMOOTHING_WINDOW:-48}
TARGET_SMOOTH_FLAGS=()
if [ "${TARGET_SMOOTHING_WINDOW}" -gt 1 ] 2>/dev/null; then
  TARGET_SMOOTH_FLAGS=(--target-smoothing-window "${TARGET_SMOOTHING_WINDOW}")
  echo "Pre-split target smoothing: window=${TARGET_SMOOTHING_WINDOW} (causal trailing MA on value column per CSV)."
else
  echo "Pre-split target smoothing: off (TARGET_SMOOTHING_WINDOW=${TARGET_SMOOTHING_WINDOW})."
fi

# Input feature selection. Two ways to override at submit time:
#   1) INPUT_FEATURES — semicolon-separated column names (so column names with spaces work).
#        sbatch --export=ALL,INPUT_FEATURES="Acceleration RMS;Kurtosis;Battery Voltage" run_dlinear_tuning.sbatch
#   2) USE_ALL_NUMERIC_FEATURES=1 uses every numeric column except TIMESTAMP / smoothed target.
#        Set USE_ALL_NUMERIC_FEATURES=0 (and leave INPUT_FEATURES empty) to fall back to TARGET_COLUMN only.
INPUT_FEATURES=${INPUT_FEATURES:-Acceleration RMS}
USE_ALL_NUMERIC_FEATURES=${USE_ALL_NUMERIC_FEATURES:-0}
if [ -n "${INPUT_FEATURES}" ]; then
  IFS=';' read -r -a INPUT_FEATURE_ARR <<< "${INPUT_FEATURES}"
  FEATURE_FLAGS=(--feature-columns "${INPUT_FEATURE_ARR[@]}")
  echo "Input features (explicit): ${INPUT_FEATURE_ARR[@]}"
elif [ "${USE_ALL_NUMERIC_FEATURES}" = "1" ]; then
  FEATURE_FLAGS=(--use-all-numeric-features)
  echo "Input features: all numeric columns."
else
  FEATURE_FLAGS=()
  echo "Input features: target column only (${TARGET_COLUMN})."
fi

# Dataset selection (override at submit time).
# Option A — ONE CSV + window ratios: SINGLE_CSV (+ TRAIN_RATIO, VAL_RATIO, optional TEST_RATIO).
# Option B — three files: TRAIN_CSV + VAL_CSV + TEST_CSV (explicit paths or from pre-split below).
#
# Option C — chronological row split saved to disk, then Option B:
#   1) Run split before training (writes train.csv / val.csv / test.csv):
#        sbatch --export=ALL,SPLIT_SOURCE_CSV=data/your.csv run_dlinear_tuning.sbatch
#      Optional: SPLIT_OUT_DIR=... (default: <repo>/data/splits/<stem>_r0.6_0.2_0.2/)
#      Optional ratios: CHRONO_TRAIN_RATIO, CHRONO_VAL_RATIO, CHRONO_TEST_RATIO (default 0.6 / 0.2 / 0.2)
#   2) Or split offline: python split_csv_chronological_train_val_test.py -i data/your.csv -o data/splits/my_run
#      then: sbatch --export=ALL,SPLIT_OUT_DIR=data/splits/my_run,SINGLE_CSV= run_dlinear_tuning.sbatch
#      (SINGLE_CSV must be empty when using SPLIT_OUT_DIR or three explicit CSV paths.)
# Option B (30 min cadence): use data_30mins_frequency/*_30_min.csv with REQUIRE_UNIFORM_TIMESTEP=1 (default).
# Per sensor: sbatch --export=ALL,SPLIT_SOURCE_CSV=data_30mins_frequency/AHU_2_9_Blower_NDE_V_30_min.csv ...
SPLIT_SOURCE_CSV=${SPLIT_SOURCE_CSV:-data_30mins_frequency/AHU_2_9_Blower_DE_A_30_min.csv}
SPLIT_OUT_DIR=${SPLIT_OUT_DIR:-}
CHRONO_TRAIN_RATIO=${CHRONO_TRAIN_RATIO:-0.6}
CHRONO_VAL_RATIO=${CHRONO_VAL_RATIO:-0.2}
CHRONO_TEST_RATIO=${CHRONO_TEST_RATIO:-0.2}

# Project root (must contain train_dlinear_sweep.py, split_csv_chronological_train_val_test.py, etc.).
# On Slurm the *copy* of this script often lives under .../spool/jobNNNN/ — dirname(BASH_SOURCE) is NOT your repo.
# Fix: (1) submit from the repo:  cd /path/to/Murata_NGYX && sbatch run_dlinear_tuning.sbatch  → SLURM_SUBMIT_DIR works
#      (2) or set in this file or via  sbatch --export=ALL,REPO_ROOT=/path/to/Murata_NGYX
REPO_ROOT=${REPO_ROOT:-}
if [ -n "${REPO_ROOT}" ]; then
  SCRIPT_DIR="${REPO_ROOT}"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${SCRIPT_DIR}" || {
  echo "ERROR: cannot cd to SCRIPT_DIR=${SCRIPT_DIR} (set REPO_ROOT to your Murata_NGYX clone)." >&2
  exit 1
}
echo "Using SCRIPT_DIR=${SCRIPT_DIR}"

TRAIN_CSV=${TRAIN_CSV:-}
VAL_CSV=${VAL_CSV:-}
TEST_CSV=${TEST_CSV:-}
SINGLE_CSV=${SINGLE_CSV:-}
TRAIN_RATIO=${TRAIN_RATIO:-0.80}
VAL_RATIO=${VAL_RATIO:-0.10}

if [ -n "${SPLIT_SOURCE_CSV}" ]; then
  if [ -z "${SPLIT_OUT_DIR}" ]; then
    _split_base="$(basename "${SPLIT_SOURCE_CSV}" .csv)"
    SPLIT_OUT_DIR="${SCRIPT_DIR}/data/splits/${_split_base}_r${CHRONO_TRAIN_RATIO}_${CHRONO_VAL_RATIO}_${CHRONO_TEST_RATIO}"
  fi
  mkdir -p "${SPLIT_OUT_DIR}"
  python -u "${SCRIPT_DIR}/split_csv_chronological_train_val_test.py" \
    --input "${SPLIT_SOURCE_CSV}" \
    --out-dir "${SPLIT_OUT_DIR}" \
    --train-ratio "${CHRONO_TRAIN_RATIO}" \
    --val-ratio "${CHRONO_VAL_RATIO}" \
    --test-ratio "${CHRONO_TEST_RATIO}"
  TRAIN_CSV="${SPLIT_OUT_DIR}/train.csv"
  VAL_CSV="${SPLIT_OUT_DIR}/val.csv"
  TEST_CSV="${SPLIT_OUT_DIR}/test.csv"
  SINGLE_CSV=""
elif [ -n "${SPLIT_OUT_DIR}" ] && [ -f "${SPLIT_OUT_DIR}/train.csv" ] && [ -f "${SPLIT_OUT_DIR}/val.csv" ] && [ -f "${SPLIT_OUT_DIR}/test.csv" ]; then
  TRAIN_CSV="${SPLIT_OUT_DIR}/train.csv"
  VAL_CSV="${SPLIT_OUT_DIR}/val.csv"
  TEST_CSV="${SPLIT_OUT_DIR}/test.csv"
  SINGLE_CSV=""
elif [ -n "${TRAIN_CSV}" ] && [ -n "${VAL_CSV}" ] && [ -n "${TEST_CSV}" ]; then
  SINGLE_CSV=""
fi

if [ -n "${TRAIN_CSV}" ] && [ -n "${VAL_CSV}" ] && [ -n "${TEST_CSV}" ]; then
  DATA_FLAGS="--train-csv ${TRAIN_CSV} --val-csv ${VAL_CSV} --test-csv ${TEST_CSV}"
  DATA_TAG="chrono_$(basename "$(dirname "${TRAIN_CSV}")")"
  REF_PARENT="$(dirname "${TRAIN_CSV}")"
elif [ -n "${SINGLE_CSV}" ]; then
  DATA_FLAGS="--single-csv ${SINGLE_CSV} --train-ratio ${TRAIN_RATIO} --val-ratio ${VAL_RATIO}"
  if [ -n "${TEST_RATIO:-}" ]; then
    DATA_FLAGS="${DATA_FLAGS} --test-ratio ${TEST_RATIO}"
  fi
  DATA_TAG="$(basename "${SINGLE_CSV}" .csv)_single_r${TRAIN_RATIO}_v${VAL_RATIO}"
  REF_PARENT="$(dirname "${SINGLE_CSV}")"
else
  echo "ERROR: Set one of: SINGLE_CSV, or TRAIN_CSV+VAL_CSV+TEST_CSV, or SPLIT_SOURCE_CSV / SPLIT_OUT_DIR." >&2
  exit 1
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

MODEL_SLUG="dlinear"
TRAIN_SCRIPT="train_dlinear_sweep.py"
RUN_TAG="${MODEL_SLUG}"

# DLinear-only: decomposition kernel (default 11; override via sbatch --export=ALL,KERNEL_SIZE=25)
KERNEL_SIZE=${KERNEL_SIZE:-11}

# outputs_dlinear_tuning_<data_folder>_<data_tag>[_job<id>] — job suffix matches run_transformer_tuning.sbatch
ROOT="outputs_${MODEL_SLUG}_tuning${DATA_FOLDER_SUFFIX}_${DATA_TAG}${JOB_SUFFIX}"
RUNS="${ROOT}/runs"
BEST="${ROOT}/best_model"
mkdir -p "${RUNS}" "${BEST}"

echo "Output directory root: ${ROOT}"

SWEEP_RUN_TAG="cfg_48_k11_lr5e4_wd${WEIGHT_DECAY}"

# 1) Run candidate configs and save window plots for each candidate.
python -u ${TRAIN_SCRIPT} ${COMMON_TRAIN} ${COMMON_LOSS} "${COMMON_QUANT[@]}" "${TARGET_SMOOTH_FLAGS[@]}" --value-column "${TARGET_COLUMN}" "${FEATURE_FLAGS[@]}" ${DATA_FLAGS} --save-window-plots "${STITCH_HTML_FLAGS[@]}" "${COMMON_ROLLING[@]}" --input-lens 48 --kernel-size "${KERNEL_SIZE}" --lr 5e-4 --weight-decay "${WEIGHT_DECAY}" --output-dir "${RUNS}/${SWEEP_RUN_TAG}"

# 2) Rank all candidate runs and store best config summary.
python -u select_best_sweep_run.py --runs-root "${RUNS}" --out-json "best_run_selection.json"

# 3) Re-run only the selected best candidate with window plots enabled.
if [ -n "${SINGLE_CSV}" ]; then
  RERUN_DATA=(--single-csv "${SINGLE_CSV}" --train-ratio "${TRAIN_RATIO}" --val-ratio "${VAL_RATIO}")
  if [ -n "${TEST_RATIO:-}" ]; then
    RERUN_DATA+=(--test-ratio "${TEST_RATIO}")
  fi
elif [ -n "${TRAIN_CSV}" ] && [ -n "${VAL_CSV}" ] && [ -n "${TEST_CSV}" ]; then
  RERUN_DATA=(--train-csv "${TRAIN_CSV}" --val-csv "${VAL_CSV}" --test-csv "${TEST_CSV}")
else
  echo "ERROR: Rerun step needs single-CSV mode or explicit train/val/test CSVs." >&2
  exit 1
fi
python -u rerun_best_sweep_with_plots.py \
  --selection-json "${RUNS}/best_run_selection.json" \
  --train-script "${TRAIN_SCRIPT}" \
  --output-root "${BEST}" \
  "${RERUN_DATA[@]}" \
  --value-column "${TARGET_COLUMN}" \
  "${FEATURE_FLAGS[@]}" \
  "${STITCH_HTML_FLAGS[@]}" \
  "${COMMON_ROLLING[@]}" \
  ${COMMON_TRAIN} \
  ${COMMON_LOSS} \
  --target-smoothing-window "${TARGET_SMOOTHING_WINDOW}"

python -u select_best_sweep_run.py --runs-root "${BEST}" --out-json "best_model_selection.json"

echo "DLinear tuning completed. Mode=${RUN_TAG}"
echo "Candidates ranking: ${SCRIPT_DIR}/${RUNS}/sweep_run_ranking.csv"
echo "Selected best candidate metadata: ${SCRIPT_DIR}/${RUNS}/best_run_selection.json"
echo ""
echo "=== Forecast plot paths (relative to SCRIPT_DIR above) ==="
echo "Sweep train output dir: ${SCRIPT_DIR}/${RUNS}/${SWEEP_RUN_TAG}/"
echo "Sweep per-window PNGs:  ${SCRIPT_DIR}/${RUNS}/${SWEEP_RUN_TAG}/rolling_window_forecasts/plots/"
echo "Stitched test HTML (if plotly installed): ${SCRIPT_DIR}/${RUNS}/${SWEEP_RUN_TAG}/rolling_window_forecasts/stitched_test_forecast.html"
echo "Best rerun root:        ${SCRIPT_DIR}/${BEST}/"
echo "(Python also prints 'Output directory' and 'Saved artifacts' with absolute paths in the log.)"
if command -v find >/dev/null 2>&1; then
  echo "Best rerun plot dirs found:"
  find "${SCRIPT_DIR}/${BEST}" -type d -path '*/rolling_window_forecasts/plots' 2>/dev/null | head -20 || true
fi
