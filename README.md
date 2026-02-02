# Sparse MinkowskiEngine Training

This repository trains a sparse MinkowskiEngine classifier on DUNE-style wire-plane images stored in a ROOT file. The pipeline converts dense 2D detector images from three views (U/V/W) into sparse tensors, shards the data to disk, and then trains a binary classifier with balanced mini-batches using per-plane 2D encoders and late-fusion logits.

## Quick start

```bash
# 1) Prepare shards from the ROOT file
python scripts/prepare.py

# 2) (Optional) sanity/overfit check
python scripts/overfit_check.py

# 3) Train
python scripts/train.py
```

If you need a full workflow script with environment variables, see [`workflow_updated.cmds`](workflow_updated.cmds) or the SLURM-friendly [`workflow.sh`](workflow.sh).

## Project layout

- `likelihood/data.py` — ROOT ingestion, sparse conversion, sharding, and datasets.
- `likelihood/model.py` — 2D MinkowskiEngine encoder used per plane.
- `likelihood/train.py` — training loop with balanced batches and gated-logit fusion.
- `scripts/prepare.py` — shard creation entry point.
- `scripts/overfit_check.py` — quick overfit diagnostic.

## Data pipeline (technical details)

### 1) Dense planes → sparse points
Each event provides three 2D planes (`detector_image_u`, `detector_image_v`, `detector_image_w`). For every plane:

- Flatten `(H, W)` into `flat`, with **drift along the y-axis (rows)** and **wire along the x-axis (columns)**.
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

Each sparse point stores a 3D coordinate and a 3D feature vector:

- **Coordinates:** `(view, y, x)` where `view ∈ {0,1,2}`.
- **Features:** `(occupancy, log_charge, view_id)` with `view_id ∈ {-1, 0, +1}`.

### 2) Sharding
Events are grouped into shards on disk to keep training I/O predictable:

- Shard size: `SHARD_EVENTS` (default 2048).
- Each shard stores concatenated coordinates/features and a `starts` array to index each event.
- Metadata in `index.pt` includes labels, weights, and non-zero counts.

### 3) Balanced batches
The training loader uses `BalancedBatchSampler` to build class-balanced batches with `BATCH/2` signal and `BATCH/2` background each step. Each epoch runs a fixed number of steps (`STEPS_PER_EPOCH`, default 1000) and resamples with a deterministic seed for reproducibility.

## Model architecture

Each plane uses a sparse 2D residual encoder with global max pooling:

- **Input features:** per-plane `(occupancy, log_charge)` only.
- **Encoder:** repeated residual blocks + strided 2D convolutions.
- **Pooling:** global max pooling.
- **Classifier head:** linear projection to a single logit.

The three per-plane logits are combined with `LateFusionClassifier` using `fusion="gated_logits"`, which learns sample-dependent weights and respects the per-plane availability mask.

## Training loop

Training (`likelihood/train.py`) uses:

- **Loss:** `BCEWithLogitsLoss` on the binary labels.
- **Optimizer:** `SGD` with momentum.
- **LR schedule:** polynomial decay.
- **Validation:** fixed number of validation batches every `VAL_EVERY` steps.

## Troubleshooting: training stuck near chance

If loss stays near ~0.693 and accuracy near ~50%, use the checks below to isolate data, label, or optimization issues quickly.

### 1) Verify sparsification isn’t collapsing inputs

If most events are empty, training degenerates to a constant bias. Use the stored `index.pt` metadata to spot this:

```python
import os, numpy as np, torch
import likelihood.config as cfg

meta = torch.load(os.path.join(cfg.SHARDS_DIR, "index.pt"), map_location="cpu")
labels = np.asarray(meta["labels"], dtype=np.uint8)
nnz = np.asarray(meta["nnz"], dtype=np.int32)

print("n_events:", len(labels))
print("class counts:", np.bincount(labels))
print("placeholder frac (nnz==1):", np.mean(nnz == 1))

for cls in [0, 1]:
    m = (labels == cls)
    q = np.quantile(nnz[m], [0, 0.5, 0.9, 0.99, 1.0])
    print(f"class {cls} nnz quantiles:", q)
```

If the placeholder fraction is large, reduce `THRESH` or enable signed log ADC (`ADC_SIGNLOG=1`) so negative hits are retained.

### 2) Check feature dynamic range in a batch

```python
import torch
import likelihood.config as cfg

coords, feats, y = next(iter(trn_loader))
nnz_per_evt = torch.bincount(coords[:, 0], minlength=cfg.BATCH).cpu()

print("batch y mean:", float(y.mean()))
print("nnz/event: min/med/max:", int(nnz_per_evt.min()), float(nnz_per_evt.median()), int(nnz_per_evt.max()))
print("frac nnz==1 in batch:", float((nnz_per_evt == 1).float().mean()))
print("ADC feat: min/mean/max:", float(feats[:, 0].min()), float(feats[:, 0].mean()), float(feats[:, 0].max()))
print("feat std (all channels):", [float(feats[:, k].std(unbiased=False)) for k in range(feats.shape[1])])
```

If `feats[:, 0]` is nearly constant or zero, the model cannot learn.

### 3) Confirm index/shard label alignment

If `index.pt` and shard labels are out of sync, the sampler will feed random labels:

```python
import os, numpy as np, torch
import likelihood.config as cfg

meta = torch.load(os.path.join(cfg.SHARDS_DIR, "index.pt"), map_location="cpu")
labels_all = np.asarray(meta["labels"], dtype=np.uint8)
shard_events = int(meta["shard_events"])

rng = np.random.default_rng(0)
for _ in range(20):
    gi = int(rng.integers(0, len(labels_all)))
    sid = gi // shard_events
    local = gi - sid * shard_events
    d = torch.load(os.path.join(cfg.SHARDS_DIR, f"shard_{sid:05d}.pt"), map_location="cpu")
    y_shard = int(d["labels"][local].item())
    y_idx = int(labels_all[gi])
    if y_shard != y_idx:
        print("MISMATCH!", gi, sid, local, y_idx, y_shard)
        break
else:
    print("OK: shard labels match index labels on samples")
```

If there is a mismatch, rebuild shards and `index.pt` together.

### 4) Prove weights update

Inside the first train step, verify that parameters change:

```python
if epoch == 0 and i == 0:
    with torch.no_grad():
        w_before = model.head[-1].weight.detach().clone()

logits = model(x).view(-1)
tloss = loss_fn(logits, y)
tloss.backward()

if micro_step % accum == 0:
    opt.step()

if epoch == 0 and i == 0:
    with torch.no_grad():
        w_after = model.head[-1].weight.detach()
        dmax = float((w_after - w_before).abs().max().item())
    print("head last layer |Δw|_max after 1 step:", dmax)
```

`dmax == 0.0` indicates the optimizer never applied an update (check LR, grad accumulation, or step execution).

### 5) Overfit a tiny fixed subset

If the model cannot overfit 64 fixed events, there is a bug in data or gradients. Use `scripts/overfit_check.py` or create a tiny fixed batch and iterate on it for a few hundred steps with dropout and weight decay disabled.

### 6) DataLoader sanity checks

- Ensure batch balance: `print("batch y mean:", float(y.mean()))` should be ~0.5 with the balanced sampler.
- If behavior changes with `NUM_WORKERS=0`, you may have worker corruption or stale caches.

### 7) Architecture-specific checks

- Monitor active sites (`x.C.shape[0]`) through the network if using stride-1 `MinkowskiConvolution`; switch to submanifold convs if nnz explodes.
- Global sum pooling scales with nnz; try average pooling or divide by count if scale varies widely.
- Check placeholder imbalance across classes (nnz==1 fraction) so the model doesn’t learn “empty” shortcuts.

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
| `STEPS_PER_EPOCH` | `1000` | Number of batches per epoch |
| `EPOCHS` | `20` | Number of epochs |
| `LR` | `3e-3` | Learning rate |
| `WEIGHT_DECAY` | `1e-4` | AdamW weight decay |
| `GRAD_CLIP` | `1.0` | Gradient clipping max norm |
| `GRAD_ACCUM_STEPS` | `1` | Gradient accumulation steps |
| `WARMUP_RATIO` | `0.02` | Warmup fraction of total steps |
| `MIN_LR_RATIO` | `0.05` | Minimum LR ratio during cosine decay |
| `NUM_WORKERS` | `8` | DataLoader workers |
| `SEED` | `12345` | Random seed |
| `VAL_FRAC` | `0.1` | Validation fraction |
| `OUT` | `checkpoint.pt` | Checkpoint path |
| `BASE_FILTERS` | `32` | Base feature width |
| `NUM_STRIDES` | `3` | Encoder/decoder depth |
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
export LR=3e-3
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
