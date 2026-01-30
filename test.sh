#!/bin/bash
set -euo pipefail

python -m unittest tests.test_features
python -m unittest tests.test_model_contracts
python -m unittest tests.test_packing_and_shards
python -m unittest tests.test_sampling_and_collate
python -m unittest tests.test_shards_roundtrip
python -m unittest tests.test_sparse_encoding
python -m unittest tests.test_train_step_mechanics
