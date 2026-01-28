#!/bin/bash
set -euo pipefail
source /gluster/data/dune/niclane/miniforge/etc/profile.d/conda.sh
conda activate /gluster/data/dune/niclane/miniforge/envs/hep-sparse-env
python -c "import torch; print(torch.cuda.is_available())"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
ROOT="/gluster/data/dune/niclane/events.root"
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  cp -u "${ROOT}" "${SLURM_TMPDIR}/events.root"
  ROOT="${SLURM_TMPDIR}/events.root"
fi
export ROOT_FILE="${ROOT}"
export OUT="checkpoint.pt"
python train.py
