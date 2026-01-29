#!/bin/bash
set -euo pipefail
source /gluster/data/dune/niclane/miniforge/etc/profile.d/conda.sh
conda activate /gluster/data/dune/niclane/miniforge/envs/hep-sparse-env
python -c "import torch; print(torch.cuda.is_available())"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export SHARDS_DIR="${SHARDS_DIR:-/gluster/data/dune/niclane/sparse_shards}"
export OUT="${OUT:-checkpoint.pt}"

if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  LOCAL="${SLURM_TMPDIR}/$(basename "${SHARDS_DIR}")"
  if [[ ! -d "${LOCAL}" ]]; then
    mkdir -p "${LOCAL}"
    rsync -a "${SHARDS_DIR}/" "${LOCAL}/"
  fi
  export SHARDS_DIR="${LOCAL}"
fi

python train.py
