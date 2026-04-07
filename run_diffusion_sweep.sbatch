#!/bin/bash
#SBATCH -J murata_diffusion_sweep
#SBATCH -A prj0000000262
#SBATCH -p h200p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=36G
#SBATCH --time=10:00:00
#SBATCH -o ~/logs/%x-%j.out
#SBATCH -e ~/logs/%x-%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$HOME/Murata_NGYX}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_DIR/outputs/diffusion_sweep/$SLURM_JOB_ID}
APPTAINER_IMAGE=${APPTAINER_IMAGE:-}

mkdir -p "$HOME/logs"
mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_DIR"

if [ -f /etc/profile.d/modules.sh ]; then
  . /etc/profile.d/modules.sh
elif [ -f /usr/share/Modules/init/bash ]; then
  . /usr/share/Modules/init/bash
fi

# Optional host environment activation.
# source "$HOME/miniconda3/etc/profile.d/conda.sh"
# conda activate torchenv
# OR
# source "$PROJECT_DIR/torchenv/bin/activate"

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Project dir: $PROJECT_DIR"
echo "Output dir: $OUTPUT_DIR"
nvidia-smi || true

if [ -n "$APPTAINER_IMAGE" ]; then
  APPTAINER_BIN="$(command -v apptainer || true)"
  [ -z "$APPTAINER_BIN" ] && APPTAINER_BIN="$(command -v singularity || true)"
  [ -z "$APPTAINER_BIN" ] && { echo "FATAL: apptainer/singularity not found"; exit 127; }
  [ -f "$APPTAINER_IMAGE" ] || { echo "FATAL: image not found: $APPTAINER_IMAGE"; exit 2; }

  BIND_OPTS="--bind $PROJECT_DIR:/workspace --bind /scratch:/scratch --bind $HOME/logs:$HOME/logs"

  srun "$APPTAINER_BIN" exec --nv $BIND_OPTS "$APPTAINER_IMAGE" bash -lc '
    set -euo pipefail
    cd /workspace
    python -u train_diffusion_sweep.py \
      --train-val-csv /workspace/data_train_val.csv \
      --test-csv /workspace/data_test_anomalous.csv \
      --output-dir "'$OUTPUT_DIR'"
  '
else
  python -u train_diffusion_sweep.py \
    --train-val-csv "$PROJECT_DIR/data_train_val.csv" \
    --test-csv "$PROJECT_DIR/data_test_anomalous.csv" \
    --output-dir "$OUTPUT_DIR"
fi

EXIT_CODE=$?
echo "End: $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
