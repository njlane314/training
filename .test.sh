#!/usr/bin/env bash
set -euo pipefail

python -m unittest tests.test_features -v
python -m unittest tests.test_model_contracts -v
python -m unittest tests.test_packing_and_shards -v
python -m unittest tests.test_sampling_and_collate -v
python -m unittest tests.test_shards_roundtrip -v
python -m unittest tests.test_sparse_encoding -v
python -m unittest tests.test_train_step_mechanics -v
