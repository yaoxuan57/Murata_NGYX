#!/bin/bash
#SBATCH -A prj0000000262
#SBATCH -p testqueue
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 5:00:00
#SBATCH -J dlinear_tuning
#SBATCH -o ~/logs/%x_%j.out
#SBATCH -e ~/logs/%x_%j.err

set -euo pipefail

mkdir -p ~/logs

COMMON_TRAIN="--epochs 250 --batch-size 16 --early-stopping-patience 20 --scheduler-patience 6 --pred-lens 288"
COMMON_LOSS="--loss-point-weight 0.7 --loss-diff-weight 0.9 --loss-curvature-weight 0.5 --loss-variance-weight 0.2"
# Per-window rolling CSV/PNG under rolling_window_forecasts/ (full series still in all_windows_forecasts.csv).
ROLLING_WINDOW_ARTIFACT_LIMIT=${ROLLING_WINDOW_ARTIFACT_LIMIT:-20}
COMMON_ROLLING="--rolling-window-artifact-limit ${ROLLING_WINDOW_ARTIFACT_LIMIT}"

# Dataset selection (override at submit time).
# CSVs must include TIMESTAMP and the target column (default: Acceleration RMS (smoothed); override with --value-column).
TRAIN_VAL_CSV=${TRAIN_VAL_CSV:-data_train_val.csv}
TEST_CSV=${TEST_CSV:-data_test_anomalous.csv}
TRAIN_VAL_BASENAME="$(basename "${TRAIN_VAL_CSV}" .csv)"
TEST_BASENAME="$(basename "${TEST_CSV}" .csv)"
DATA_TAG="${TRAIN_VAL_BASENAME}_to_${TEST_BASENAME}"
# Include parent folder of train CSV in output root when not the current dir (e.g. AHU 2-9 Blower DE A).
TRAIN_VAL_PARENT="$(dirname "${TRAIN_VAL_CSV}")"
DATA_FOLDER_NAME="$(basename "${TRAIN_VAL_PARENT}")"
if [ "${DATA_FOLDER_NAME}" = "." ]; then
  DATA_FOLDER_SUFFIX=""
else
  DATA_FOLDER_SUFFIX="_${DATA_FOLDER_NAME}"
fi
# Toggle MoE mode from submission command:
#   sbatch --export=USE_MOE=1,MOE_NUM_EXPERTS=4 run_dlinear_tuning.sbatch
# Default is plain DLinear (USE_MOE=0).
USE_MOE=${USE_MOE:-0}
MOE_NUM_EXPERTS=${MOE_NUM_EXPERTS:-4}
MOE_EXPERTS_LIST=${MOE_EXPERTS_LIST:-${MOE_NUM_EXPERTS}}

if [ "${USE_MOE}" = "1" ]; then
  TRAIN_SCRIPT="train_moe_dlinear_sweep.py"
  DATA_FLAGS="--train-val-csv ${TRAIN_VAL_CSV} --test-csv ${TEST_CSV}"

  for MOE_NUM_EXPERTS in ${MOE_EXPERTS_LIST}; do
    EXTRA_MODEL_ARGS="--num-experts ${MOE_NUM_EXPERTS}"
    RUN_TAG="moe_k${MOE_NUM_EXPERTS}"

    ROOT="outputs_dlinear_tuning_${RUN_TAG}${DATA_FOLDER_SUFFIX}_${DATA_TAG}"
    RUNS="${ROOT}/runs"
    BEST="${ROOT}/best_model"
    mkdir -p "${RUNS}" "${BEST}"

    echo "Running MoE tuning with ${MOE_NUM_EXPERTS} experts -> ${RUN_TAG}"

    # 1) Run candidate configs and save window plots for each candidate.
    python -u ${TRAIN_SCRIPT} ${COMMON_TRAIN} ${COMMON_LOSS} ${EXTRA_MODEL_ARGS} ${DATA_FLAGS} --save-window-plots ${COMMON_ROLLING} --input-lens 864 --kernel-size 11 --lr 5e-4 --weight-decay 0 --output-dir "${RUNS}/cfg_864_k11_lr5e4_moe${MOE_NUM_EXPERTS}"
    python -u ${TRAIN_SCRIPT} ${COMMON_TRAIN} ${COMMON_LOSS} ${EXTRA_MODEL_ARGS} ${DATA_FLAGS} --save-window-plots ${COMMON_ROLLING} --input-lens 1296 --kernel-size 11 --lr 2e-4 --weight-decay 1e-6 --output-dir "${RUNS}/cfg_1296_k11_lr2e4_moe${MOE_NUM_EXPERTS}"
    python -u ${TRAIN_SCRIPT} ${COMMON_TRAIN} ${COMMON_LOSS} ${EXTRA_MODEL_ARGS} ${DATA_FLAGS} --save-window-plots ${COMMON_ROLLING} --input-lens 1728 --kernel-size 11 --lr 2e-4 --weight-decay 1e-6 --output-dir "${RUNS}/cfg_1728_k11_lr2e4_moe${MOE_NUM_EXPERTS}"
    python -u ${TRAIN_SCRIPT} ${COMMON_TRAIN} ${COMMON_LOSS} ${EXTRA_MODEL_ARGS} ${DATA_FLAGS} --save-window-plots ${COMMON_ROLLING} --input-lens 2160 --kernel-size 11 --lr 2e-4 --weight-decay 1e-6 --output-dir "${RUNS}/cfg_2160_k11_lr2e4_moe${MOE_NUM_EXPERTS}"
    python -u ${TRAIN_SCRIPT} ${COMMON_TRAIN} ${COMMON_LOSS} ${EXTRA_MODEL_ARGS} ${DATA_FLAGS} --save-window-plots ${COMMON_ROLLING} --input-lens 2592 --kernel-size 11 --lr 2e-4 --weight-decay 1e-6 --output-dir "${RUNS}/cfg_2592_k11_lr2e4_moe${MOE_NUM_EXPERTS}"

    # 2) Rank all candidate runs and store best config summary.
    python -u select_best_sweep_run.py --runs-root "${RUNS}" --out-json "best_run_selection.json"

    # 3) Re-run only the selected best candidate with window plots enabled.
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

    echo "MoE tuning completed. Mode=${RUN_TAG}"
    echo "Candidates ranking: ${RUNS}/sweep_run_ranking.csv"
    echo "Selected best candidate metadata: ${RUNS}/best_run_selection.json"
    echo "Candidate run plots under: ${RUNS}/<cfg>/rolling_window_forecasts/plots"
    echo "Best-model-only window plots under: ${BEST}/<best_cfg>/rolling_window_forecasts/plots"
  done
else
  TRAIN_SCRIPT="train_dlinear_sweep.py"
  RUN_TAG="dlinear"
  DATA_FLAGS="--train-val-csv ${TRAIN_VAL_CSV} --test-csv ${TEST_CSV}"

  ROOT="outputs_dlinear_tuning_${RUN_TAG}${DATA_FOLDER_SUFFIX}_${DATA_TAG}"
  RUNS="${ROOT}/runs"
  BEST="${ROOT}/best_model"
  mkdir -p "${RUNS}" "${BEST}"

  # 1) Run candidate configs and save window plots for each candidate.
  python -u ${TRAIN_SCRIPT} ${COMMON_TRAIN} ${COMMON_LOSS} ${DATA_FLAGS} --save-window-plots ${COMMON_ROLLING} --input-lens 864 --kernel-size 11 --lr 5e-4 --weight-decay 0 --output-dir "${RUNS}/cfg_864_k11_lr5e4"
  python -u ${TRAIN_SCRIPT} ${COMMON_TRAIN} ${COMMON_LOSS} ${DATA_FLAGS} --save-window-plots ${COMMON_ROLLING} --input-lens 1296 --kernel-size 11 --lr 5e-4 --weight-decay 0 --output-dir "${RUNS}/cfg_1296_k11_lr5e4"
  python -u ${TRAIN_SCRIPT} ${COMMON_TRAIN} ${COMMON_LOSS} ${DATA_FLAGS} --save-window-plots ${COMMON_ROLLING} --input-lens 1728 --kernel-size 11 --lr 2e-4 --weight-decay 1e-6 --output-dir "${RUNS}/cfg_1728_k11_lr2e4"
  python -u ${TRAIN_SCRIPT} ${COMMON_TRAIN} ${COMMON_LOSS} ${DATA_FLAGS} --save-window-plots ${COMMON_ROLLING} --input-lens 2160 --kernel-size 11 --lr 2e-4 --weight-decay 1e-6 --output-dir "${RUNS}/cfg_2160_k11_lr2e4"
  python -u ${TRAIN_SCRIPT} ${COMMON_TRAIN} ${COMMON_LOSS} ${DATA_FLAGS} --save-window-plots ${COMMON_ROLLING} --input-lens 2592 --kernel-size 11 --lr 2e-4 --weight-decay 1e-6 --output-dir "${RUNS}/cfg_2592_k11_lr2e4"

  # 2) Rank all candidate runs and store best config summary.
  python -u select_best_sweep_run.py --runs-root "${RUNS}" --out-json "best_run_selection.json"

  # 3) Re-run only the selected best candidate with window plots enabled.
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

  echo "DLinear tuning completed. Mode=${RUN_TAG}"
  echo "Candidates ranking: ${RUNS}/sweep_run_ranking.csv"
  echo "Selected best candidate metadata: ${RUNS}/best_run_selection.json"
  echo "Candidate run plots under: ${RUNS}/<cfg>/rolling_window_forecasts/plots"
  echo "Best-model-only window plots under: ${BEST}/<best_cfg>/rolling_window_forecasts/plots"
fi
