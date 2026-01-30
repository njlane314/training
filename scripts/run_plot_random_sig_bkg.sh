#!/usr/bin/env bash
set -euo pipefail

python3 "$(dirname "$0")/plot_random_sig_bkg.py" --source shards --shards-dir "${SHARDS_DIR:-}" --skip-placeholder --out rand_shards.png
