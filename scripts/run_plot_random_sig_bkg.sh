#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "${script_dir}/plot_random_sig_bkg.py" \
  --source shards \
  --shards-dir "${SHARDS_DIR:-}" \
  --skip-placeholder \
  --out rand_shards.png
