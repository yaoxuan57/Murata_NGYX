#!/bin/bash
#SBATCH --job-name=murata_tf_sweep
#SBATCH --account=prj0000000262
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$HOME/Murata_NGYX}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_DIR/outputs/transformer_sweep/$SLURM_JOB_ID}

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_DIR"

# Choose ONE environment activation method and adjust to your server setup.
# module load cuda/12.1
# source "$HOME/miniconda3/etc/profile.d/conda.sh"
# conda activate torchenv
# OR
# source "$PROJECT_DIR/torchenv/bin/activate"

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Project dir: $PROJECT_DIR"
echo "Output dir: $OUTPUT_DIR"

python -u train_transformer_sweep.py \
  --train-val-csv "$PROJECT_DIR/data_train_val.csv" \
  --test-csv "$PROJECT_DIR/data_test_anomalous.csv" \
  --output-dir "$OUTPUT_DIR"

EXIT_CODE=$?
echo "End: $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
