# Sparse MinkowskiEngine Training

This repository trains a sparse 3D MinkowskiEngine U-Net classifier on DUNE-style wire-plane images stored in a ROOT file. The pipeline converts dense 2D detector images from three views (U/V/W) into sparse tensors, shards the data to disk, and then trains a binary classifier with balanced mini-batches.

## Quick start

```bash
# 1) Prepare shards from the ROOT file
python scripts/prepare_shards.py

# 2) (Optional) sanity/overfit check
python scripts/overfit_check.py

# 3) Train
python train.py
```

If you need a full workflow script with environment variables, see [`workflow_updated.cmds`](workflow_updated.cmds) or the SLURM-friendly [`workflow.sh`](workflow.sh).

## Project layout

- `likelihood/data.py` — ROOT ingestion, sparse conversion, sharding, and datasets.
- `likelihood/model.py` — MinkowskiEngine U-Net classifier.
- `likelihood/train.py` — training loop with balanced batches and EMA metrics.
- `scripts/prepare_shards.py` — shard creation entry point.
- `scripts/overfit_check.py` — quick overfit diagnostic.

## Data pipeline (technical details)

### 1) Dense planes → sparse points
Each event provides three 2D planes (`detector_image_u`, `detector_image_v`, `detector_image_w`). For every plane:

- Flatten `(H, W)` into `flat`.
- Threshold and transform ADC values into a sparse set of points.

Two ADC encodings are supported (controlled by `ADC_SIGNLOG`):

- **Unsigned log:**
  \[
  \text{adc} = \log(1 + \max(v, 0))
  \]
  with indices where `v > THRESH`.

- **Signed log (if `ADC_SIGNLOG=1`):**
  \[
  \text{adc} = \operatorname{sign}(v)\,\log(1 + |v|)
  \]
  with indices where `|v| > THRESH`.

Each sparse point stores a 3D coordinate and a 4D feature vector:

- **Coordinates:** `(view, y, x)` where `view ∈ {0,1,2}`.
- **Features:** `(adc, y_norm, x_norm, view_norm)` with
  \[
  y_\text{norm} = \frac{y - H/2}{H/2},\quad x_\text{norm} = \frac{x - W/2}{W/2},\quad view_\text{norm} = view - 1.
  \]

### 2) Sharding
Events are grouped into shards on disk to keep training I/O predictable:

- Shard size: `SHARD_EVENTS` (default 2048).
- Each shard stores concatenated coordinates/features and a `starts` array to index each event.
- Metadata in `index.pt` includes labels, weights, and non-zero counts.

### 3) Balanced batches
The training loader uses `BalancedBatchSampler` to build class-balanced batches with `BATCH/2` signal and `BATCH/2` background each step. Each epoch resamples with a deterministic seed for reproducibility.

## Model architecture

`MinkUNetClassifier` is a sparse 3D U-Net with residual blocks and global pooling:

- **Input normalization:** learnable shift and log-scale per feature channel.
- **Encoder:** repeated residual blocks + strided convolutions.
- **Decoder:** transposed convolutions with skip connections.
- **Pooling:** global sum + max pooling.
- **Classifier head:** MLP on pooled features plus a `log1p` count feature:
  \[
  z = [\text{sum\_pool}(x),\ \text{max\_pool}(x),\ \log(1 + N_{\text{points}})]
  \]
  then linear layers to produce a single logit.

## Training loop

Training (`likelihood/train.py`) uses:

- **Loss:** `BCEWithLogitsLoss` on the binary labels.
- **Optimizer:** `AdamW`.
- **LR schedule:** `OneCycleLR`.
- **Validation:** a random probe batch from the validation set each step to track EMA of loss.
- **Checkpoint:** best EMA validation loss (`checkpoint.pt` by default).

## Configuration (environment variables)

`likelihood/config.py` reads all configuration from environment variables. Key settings include:

| Variable | Default | Purpose |
| --- | --- | --- |
| `ROOT_FILE` | `/gluster/data/dune/niclane/events.root` | Input ROOT file |
| `TREE` | `events` | ROOT tree name |
| `SHARDS_DIR` | `/gluster/data/dune/niclane/sparse_shards` | Shard input directory |
| `SHARDS_OUT` | `SHARDS_DIR` | Shard output directory |
| `H`, `W` | `512` | Image height/width |
| `THRESH` | `0.0` | ADC threshold |
| `ADC_SIGNLOG` | `0` | Signed-log ADC transform |
| `SHARD_EVENTS` | `2048` | Events per shard |
| `CHUNK_EVENTS` | `64` | ROOT read chunk size |
| `BATCH` | `32` | Batch size (must be even) |
| `EPOCHS` | `20` | Number of epochs |
| `LR` | `3e-4` | Peak learning rate |
| `WEIGHT_DECAY` | `1e-4` | AdamW weight decay |
| `NUM_WORKERS` | `8` | DataLoader workers |
| `SEED` | `12345` | Random seed |
| `VAL_FRAC` | `0.1` | Validation fraction |
| `OUT` | `checkpoint.pt` | Checkpoint path |
| `BASE_FILTERS` | `32` | Base feature width |
| `NUM_STRIDES` | `4` | Encoder/decoder depth |
| `DROPOUT` | `0.2` | Dropout probability |

### Example configuration

```bash
export ROOT_FILE=/data/events.root
export TREE=events
export SHARDS_DIR=/data/sparse_shards
export SHARDS_OUT=$SHARDS_DIR
export H=512
export W=512
export THRESH=0.0
export ADC_SIGNLOG=0
export BATCH=32
export EPOCHS=20
export LR=3e-4
export OUT=checkpoint.pt
```

## Workflow notes

- **Shard creation:** required before training; it writes `index.pt` plus `shard_*.pt` files.
- **Overfit check:** uses a small subset to verify the model can learn.
- **Training:** prints per-step loss/accuracy and saves the best EMA validation checkpoint.
- **SLURM usage:** `workflow.sh` syncs shards to `$SLURM_TMPDIR` for local I/O.

## Dependencies

- Python, PyTorch
- [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)
- uproot
- numpy

Install dependencies in your preferred environment, then follow the workflow above.
