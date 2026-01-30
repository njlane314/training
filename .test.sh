#!/usr/bin/env bash
set -euo pipefail

cat <<'INFO'
Practical workflow: how to use these tests to debug systematically

Run only encoding tests:
python -m unittest tests.test_sparse_encoding -v
If these fail, the rest of the pipeline is downstream noise.

Run pack/dataset tests:
python -m unittest tests.test_packing_and_shards -v
If these fail, training instability may be I/O / slicing, not ML.

Run sampler/collate tests:
python -m unittest tests.test_sampling_and_collate -v
If these fail, MinkowskiEngine inputs are malformed (most frequent source of “it trains but nonsense”).

Run model contracts (on a box with MinkowskiEngine):
python -m unittest tests.test_model_contracts -v
INFO

python -m unittest tests.test_sparse_encoding -v
python -m unittest tests.test_packing_and_shards -v
python -m unittest tests.test_sampling_and_collate -v
python -m unittest tests.test_model_contracts -v
