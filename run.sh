#!/bin/bash

# ==============================================================================
# ## --- Environment Setup --- ##
# ==============================================================================

echo "Activating Conda environment..."
source /gluster/data/dune/niclane/miniforge/etc/profile.d/conda.sh
conda activate /gluster/data/dune/niclane/miniforge/envs/hep-sparse-env

python -c "import torch; print('cuda_available=', torch.cuda.is_available())"


# ==============================================================================
# ## --- Configuration (env vars consumed by likelihood/config.py) --- ##
# ==============================================================================

export ROOT_FILE="/gluster/data/dune/niclane/events.root"
export TREE="events"

export SHARDS_DIR="/gluster/data/dune/niclane/sparse_shards"
export SHARDS_OUT="${SHARDS_DIR}"

export H=512
export W=512
export THRESH=0.0
export ADC_SIGNLOG=0

export SHARD_EVENTS=2048
export CHUNK_EVENTS=64

export BATCH=32
export EPOCHS=20
export LR=3e-4
export WEIGHT_DECAY=1e-4
export NUM_WORKERS=8
export SEED=12345
export VAL_FRAC=0.1
export OUT="checkpoint.pt"

export BASE_FILTERS=32
export NUM_STRIDES=4
export DROPOUT=0.2


# ==============================================================================
# ## --- Create Shards --- ##
# ============================================================================== 

python scripts/prepare_shards.py


# ==============================================================================
# ## --- Run Tests --- ##
# ============================================================================== 

python -m pytest -v


# ==============================================================================
# ## --- Overfit Check --- ##
# ============================================================================== 

python scripts/overfit_check.py


# ==============================================================================
# ## --- Train --- ##
# ============================================================================== 

python train.py
