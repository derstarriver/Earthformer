#!/bin/bash
# Click-to-run launcher for ENSO/SST training
# Usage: ./run_enso.sh

set -e

# ---- user config ----
GPUS=1
EXP_NAME="enso_exp"
DATA_DIR="./datasets/enso_multivar"
CFG_FILE="$(dirname "$0")/cfg.yaml"

# ---- activate conda ----
if [ -f "$HOME/.conda/etc/profile.d/conda.sh" ]; then
    source "$HOME/.conda/etc/profile.d/conda.sh"
    conda activate gxy
elif [ -n "$CONDA_PREFIX" ]; then
    echo "Using current conda env: $(basename $CONDA_PREFIX)"
else
    echo "WARNING: conda not found, using system python"
fi

# ---- run ----
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(dirname "$DIR")/../.."   # go to Earthformer root

echo "============================================"
echo "ENSO/SST Training Launcher"
echo "  GPUs:      $GPUS"
echo "  Experiment: $EXP_NAME"
echo "  Data:       $DATA_DIR"
echo "  Config:     $CFG_FILE"
echo "============================================"

python scripts/cuboid_transformer/enso/train_cuboid_enso.py \
    --gpus $GPUS \
    --save "$EXP_NAME" \
    --data_dir "$DATA_DIR" \
    --cfg "$CFG_FILE"
