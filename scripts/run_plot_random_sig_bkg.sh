#!/usr/bin/env bash
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

seed="${PLOT_SEED:-$(date +%s%N)}"

python3 "${script_dir}/plot_random_sig_bkg.py" \
  --source shards \
  --shards-dir "${SHARDS_DIR:-}" \
  --skip-placeholder \
  --seed "${seed}" \
  --out rand_shards.png
