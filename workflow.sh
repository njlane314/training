#!/bin/bash

# ==============================================================================
# ## --- Environment Setup --- ##
# ==============================================================================

echo "Activating Conda environment..."
source /gluster/data/dune/niclane/miniforge/etc/profile.d/conda.sh
conda activate /gluster/data/dune/niclane/miniforge/envs/hep-sparse-env

# Confirm CUDA availability matches the expected workflow.
python -c "import torch; print(torch.cuda.is_available())"


# ==============================================================================
# ## --- Configuration (train.py) --- ##
# ==============================================================================

ROOT_FILE="/gluster/data/dune/niclane/events.root"
TREE_NAME="events"
PYTHON_SCRIPT="train.py"
OUTPUT_WEIGHTS="./checkpoint.pt"

# --- Image Shape ---
HEIGHT=512
WIDTH=512

# --- Training Hyperparameters ---
LEARNING_RATE=0.0001
WEIGHT_DECAY=1e-4
BATCH_SIZE=32
EPOCHS=20
VAL_FRAC=0.1
THRESHOLD=0.0
DEVICE="cuda"
SEED=12345


# ==============================================================================
# ## --- EXECUTION BLOCK --- ##
# ==============================================================================

echo "Starting MinkUNet training..."
python "${PYTHON_SCRIPT}" \
  --input "${ROOT_FILE}" \
  --tree "${TREE_NAME}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LEARNING_RATE}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --val-frac "${VAL_FRAC}" \
  --threshold "${THRESHOLD}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --out "${OUTPUT_WEIGHTS}"

echo "Training complete. Model weights saved to ${OUTPUT_WEIGHTS}"
