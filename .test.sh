#!/usr/bin/env bash

python -m unittest tests.test_sparse_encoding -v
python -m unittest tests.test_packing_and_shards -v
python -m unittest tests.test_sampling_and_collate -v
python -m unittest tests.test_model_contracts -v
