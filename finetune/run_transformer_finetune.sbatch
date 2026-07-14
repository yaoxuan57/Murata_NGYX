#!/bin/bash
#SBATCH -A prj0000000262
#SBATCH -p testqueue
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -J transformer_finetune
#SBATCH -o ~/logs/%x_%j.out
#SBATCH -e ~/logs/%x_%j.err

# Fine-tune on Jan–Jun with chronological 70/15/15 row split (train → val → test in time),
# load init .pth weights.
#
# Prereqs:
#   1) python finetune/build_jan2jun_only_splits.py
#      → finetune/data_AHU_2_9_Blower_DE_A/splits/{train.csv,val.csv,test.csv}
#   2) INIT_CHECKPOINT: same architecture (input_len=48, pred_len=48).
#
# Submit from repo root:
#   cd /path/to/Murata_NGYX
#   sbatch finetune/run_transformer_finetune.sbatch
#
# With explicit checkpoint (cluster path):
#   sbatch --export=ALL,INIT_CHECKPOINT=/path/to/best_model/.../transformer_delta_huber_date_split_best.pth \
#     finetune/run_transformer_finetune.sbatch
#
# Optional overrides:
#   FINETUNE_LR=1e-5  FINETUNE_EPOCHS=80  EARLY_STOPPING_PATIENCE=15
#   REQUIRE_UNIFORM_TIMESTEP=1
#   MAX_CONSECUTIVE_TIMESTAMP_GAP_SECONDS=1860  (strict; like full tuning)

set -euo pipefail

mkdir -p ~/logs

# --- Window rules (finetune defaults: relaxed so short val/test slices get windows) ---
# MAX_CONSECUTIVE_TIMESTAMP_GAP_SECONDS:
#   (empty)|none|0  = OFF — no max-gap filter (recommended for finetune val/test)
#   1860            = same as full tuning (~31 min; often yields val/test 0 windows)
REQUIRE_UNIFORM_TIMESTEP=${REQUIRE_UNIFORM_TIMESTEP:-0}
MAX_CONSECUTIVE_TIMESTAMP_GAP_SECONDS=${MAX_CONSECUTIVE_TIMESTAMP_GAP_SECONDS:-}
UNIFORM_STEP_SECONDS=${UNIFORM_STEP_SECONDS:-1800}
UNIFORM_TOL_SECONDS=${UNIFORM_TOL_SECONDS:-60}
if [ "${REQUIRE_UNIFORM_TIMESTEP}" = "1" ]; then
  _UNIFORM_FLAGS="--require-uniform-timestep --uniform-step-seconds ${UNIFORM_STEP_SECONDS} --uniform-step-tolerance-seconds ${UNIFORM_TOL_SECONDS}"
else
  _UNIFORM_FLAGS="--no-require-uniform-timestep"
fi
_MAX_GAP="${MAX_CONSECUTIVE_TIMESTAMP_GAP_SECONDS}"
_MAX_GAP_FLAGS=()
if [ -n "${_MAX_GAP}" ] && [ "${_MAX_GAP}" != "none" ] && [ "${_MAX_GAP}" != "0" ]; then
  _MAX_GAP_FLAGS=(--max-consecutive-timestamp-gap-seconds "${_MAX_GAP}")
  echo "Max TIMESTAMP gap filter: ≤ ${_MAX_GAP}s inside each window."
else
  echo "Max TIMESTAMP gap filter: OFF (all sliding windows allowed; gaps may appear inside windows)."
fi

FINETUNE_EPOCHS=${FINETUNE_EPOCHS:-120}
FINETUNE_LR=${FINETUNE_LR:-5e-5}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-20}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
LOSS_TAIL_WEIGHT=${LOSS_TAIL_WEIGHT:-1.0}

COMMON_TRAIN="--epochs ${FINETUNE_EPOCHS} --batch-size 16 --early-stopping-patience ${EARLY_STOPPING_PATIENCE} --min-delta 1e-6 --scheduler-patience 6 --pred-lens 48 --pred-smoothing-window 1 ${_UNIFORM_FLAGS} ${_MAX_GAP_FLAGS[*]} --train-window-stride 1 --val-window-stride 0 --test-window-stride 0"
COMMON_LOSS="--loss-point-weight 0.7 --loss-diff-weight 0.9 --loss-curvature-weight 0.5 --loss-variance-weight 0.2 --loss-tail-weight ${LOSS_TAIL_WEIGHT}"

ENABLE_FORECAST_QUANTILES=${ENABLE_FORECAST_QUANTILES:-1}
COMMON_QUANT=()
if [ "${ENABLE_FORECAST_QUANTILES}" != "0" ] && [ "${ENABLE_FORECAST_QUANTILES}" != "no" ]; then
  COMMON_QUANT=(--forecast-quantiles 0.05 0.5 0.95)
fi

TARGET_COLUMN=${TARGET_COLUMN:-"Acceleration RMS"}
TARGET_SMOOTHING_WINDOW=${TARGET_SMOOTHING_WINDOW:-48}
TARGET_SMOOTH_FLAGS=()
if [ "${TARGET_SMOOTHING_WINDOW}" -gt 1 ] 2>/dev/null; then
  TARGET_SMOOTH_FLAGS=(--target-smoothing-window "${TARGET_SMOOTHING_WINDOW}")
fi

INPUT_FEATURES=${INPUT_FEATURES:-Acceleration RMS}
USE_ALL_NUMERIC_FEATURES=${USE_ALL_NUMERIC_FEATURES:-0}
if [ -n "${INPUT_FEATURES}" ]; then
  IFS=';' read -r -a INPUT_FEATURE_ARR <<< "${INPUT_FEATURES}"
  FEATURE_FLAGS=(--feature-columns "${INPUT_FEATURE_ARR[@]}")
elif [ "${USE_ALL_NUMERIC_FEATURES}" = "1" ]; then
  FEATURE_FLAGS=(--use-all-numeric-features)
else
  FEATURE_FLAGS=()
fi

SAVE_STITCHED_TEST_HTML=${SAVE_STITCHED_TEST_HTML:-1}
STITCH_HTML_FLAGS=()
if [ "${SAVE_STITCHED_TEST_HTML}" != "0" ] && [ "${SAVE_STITCHED_TEST_HTML}" != "no" ]; then
  STITCH_HTML_FLAGS=(--save-stitched-test-html)
fi

ROLLING_WINDOW_ARTIFACT_LIMIT=${ROLLING_WINDOW_ARTIFACT_LIMIT:-}
COMMON_ROLLING=()
if [ -n "${ROLLING_WINDOW_ARTIFACT_LIMIT}" ] && [ "${ROLLING_WINDOW_ARTIFACT_LIMIT}" -gt 0 ] 2>/dev/null; then
  COMMON_ROLLING=(--rolling-window-artifact-limit "${ROLLING_WINDOW_ARTIFACT_LIMIT}")
fi

# --- Repo root (must contain train_transformer_sweep.py at top level) ---
# If you submit from finetune/ (cd finetune && sbatch ...), SLURM_SUBMIT_DIR is wrong unless we fix it.
_FINETUNE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_DEFAULT_REPO="$(cd "${_FINETUNE_DIR}/.." && pwd)"

REPO_ROOT=${REPO_ROOT:-}
if [ -n "${REPO_ROOT}" ]; then
  SCRIPT_DIR="${REPO_ROOT}"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/train_transformer_sweep.py" ]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
elif [ -f "${_DEFAULT_REPO}/train_transformer_sweep.py" ]; then
  SCRIPT_DIR="${_DEFAULT_REPO}"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="${_DEFAULT_REPO}"
fi
cd "${SCRIPT_DIR}" || { echo "ERROR: cannot cd to SCRIPT_DIR=${SCRIPT_DIR}" >&2; exit 1; }
if [ ! -f "${SCRIPT_DIR}/train_transformer_sweep.py" ]; then
  echo "ERROR: ${SCRIPT_DIR}/train_transformer_sweep.py not found." >&2
  echo "Submit from repo root:  cd .../Murata_NGYX && sbatch finetune/run_transformer_finetune.sbatch" >&2
  echo "Or set:  sbatch --export=ALL,REPO_ROOT=/path/to/Murata_NGYX finetune/run_transformer_finetune.sbatch" >&2
  exit 1
fi
echo "Using SCRIPT_DIR=${SCRIPT_DIR}"

# --- Chronological 70/15/15 (build_jan2jun_only_splits.py): first 70% train, next 15% val, last 15% test ---
SPLIT_DIR="${SCRIPT_DIR}/finetune/data_AHU_2_9_Blower_DE_A/splits"
TRAIN_CSV=${TRAIN_CSV:-${SPLIT_DIR}/train.csv}
VAL_CSV=${VAL_CSV:-${SPLIT_DIR}/val.csv}
TEST_CSV=${TEST_CSV:-${SPLIT_DIR}/test.csv}

# Optionally (re)build chrono splits before training.
BUILD_SPLITS=${BUILD_SPLITS:-1}
if [ "${BUILD_SPLITS}" = "1" ]; then
  SOURCE_CSV=${SOURCE_CSV:-${SCRIPT_DIR}/finetune/data_AHU_2_9_Blower_DE_A/AHU_2_9_Blower_DE_A_jan2_jun_30_min.csv}
  python -u "${SCRIPT_DIR}/finetune/build_jan2jun_only_splits.py" \
    --source "${SOURCE_CSV}" \
    --out-dir "${SPLIT_DIR}"
fi

for f in "${TRAIN_CSV}" "${VAL_CSV}" "${TEST_CSV}"; do
  if [ ! -f "${f}" ]; then
    echo "ERROR: missing split file: ${f}" >&2
    echo "Run: python finetune/build_jan2jun_only_splits.py" >&2
    exit 1
  fi
done

DATA_FLAGS="--train-csv ${TRAIN_CSV} --val-csv ${VAL_CSV} --test-csv ${TEST_CSV}"
DATA_TAG="finetune_blower_de_a_jan2jun_chrono_70_15_15"
echo "Data split: chronological train/val/test CSVs under ${SPLIT_DIR}"

# --- Prior transformer checkpoint (required) ---
INIT_CHECKPOINT=${INIT_CHECKPOINT:-${SCRIPT_DIR}/finetune/model_to_finetune/AHU_2_9_Blower_DE_A_v2.pth}
if [ ! -f "${INIT_CHECKPOINT}" ]; then
  echo "ERROR: INIT_CHECKPOINT not found: ${INIT_CHECKPOINT}" >&2
  echo "Set INIT_CHECKPOINT to the .pth from your run_transformer_tuning.sbatch best_model rerun, e.g.:" >&2
  echo "  outputs_transformer_tuning_.../best_model/cfg_48_k11_lr5e4_wd.../transformer_delta_huber_date_split_best.pth" >&2
  exit 1
fi
echo "Init checkpoint: ${INIT_CHECKPOINT}"
echo "Fine-tune LR=${FINETUNE_LR} epochs=${FINETUNE_EPOCHS} patience=${EARLY_STOPPING_PATIENCE}"

if [ -n "${SLURM_JOB_ID:-}" ]; then
  JOB_SUFFIX="_job${SLURM_JOB_ID}"
else
  JOB_SUFFIX=""
fi

ROOT="finetune/outputs_transformer_finetune_${DATA_TAG}${JOB_SUFFIX}"
mkdir -p "${ROOT}"
echo "Output directory: ${SCRIPT_DIR}/${ROOT}"

python -u "${SCRIPT_DIR}/train_transformer_sweep.py" \
  ${COMMON_TRAIN} \
  ${COMMON_LOSS} \
  "${COMMON_QUANT[@]}" \
  "${TARGET_SMOOTH_FLAGS[@]}" \
  --value-column "${TARGET_COLUMN}" \
  "${FEATURE_FLAGS[@]}" \
  ${DATA_FLAGS} \
  --save-window-plots \
  "${STITCH_HTML_FLAGS[@]}" \
  "${COMMON_ROLLING[@]}" \
  --input-lens 48 \
  --lr "${FINETUNE_LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --init-checkpoint "${INIT_CHECKPOINT}" \
  --output-dir "${ROOT}" \
  --checkpoint-name transformer_finetune_blower_de_a_best.pth

echo ""
echo "Fine-tune completed."
echo "Checkpoint: ${SCRIPT_DIR}/${ROOT}/transformer_finetune_blower_de_a_best.pth"
echo "Plots:      ${SCRIPT_DIR}/${ROOT}/rolling_window_forecasts/plots/"
