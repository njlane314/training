# train.py
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional

import MinkowskiEngine as ME

from . import config as cfg
from .dataset import BalancedBatchSampler, ShardDataset, collate_me_fusion
from .fusion import MultiViewSetClassifier
from .model import make_backbone

def poly_lr(step, max_steps, lr0, power):
    t = min(step / max_steps, 1.0)
    return lr0 * (1.0 - t) ** power

def train_llr():
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta = torch.load(f"{cfg.SHARDS_DIR}/index.pt", map_location="cpu")
    n = int(meta["n_events"])
    rng = np.random.default_rng(cfg.SEED)
    labels_all = np.asarray(meta["labels"], dtype=np.uint8).reshape(-1)

    # If shards were created with process.py, index.pt contains nnz per event.
    # Training on nnz==0 events + masking all views can yield exactly-zero pooled
    # embeddings and (with perfectly balanced batches) exactly-zero gradients.
    nnz_all: Optional[np.ndarray] = None
    if "nnz" in meta and meta["nnz"] is not None:
        if isinstance(meta["nnz"], torch.Tensor):
            nnz_all = meta["nnz"].to(dtype=torch.int64).cpu().numpy().reshape(-1)
        else:
            nnz_all = np.asarray(meta["nnz"], dtype=np.int64).reshape(-1)

    if nnz_all is not None:
        if nnz_all.shape[0] != n:
            raise ValueError(f"index.pt nnz has len={nnz_all.shape[0]} but n_events={n}")

        good = nnz_all > 0
        idx_all = np.flatnonzero(good)
        dropped = int(n - idx_all.size)
        print(f"[data] keeping nnz>0 events: {idx_all.size}/{n} (dropped {dropped})")
        if idx_all.size == 0:
            raise ValueError(
                "All events have nnz==0 after sparsification. "
                "Check THRESH / branch names / shard generation (bad_events) in index.pt."
            )

        # Split only over non-empty events to guarantee at least one available view per event.
        perm = rng.permutation(idx_all.size)
        idx_perm = idx_all[perm]
        n_val = int(cfg.VAL_FRACTION * idx_perm.size)
        val_idx = idx_perm[:n_val]
        train_idx = idx_perm[n_val:]

        # BalancedBatchSampler requires both classes in the training split.
        labs_train = labels_all[train_idx]
        if labs_train.min() == labs_train.max():
            raise ValueError(
                "After filtering nnz>0, the training split contains only one class. "
                "Lower THRESH or inspect index.pt (labels/nnz/bad_events)."
            )
    else:
        perm = rng.permutation(n)
        n_val = int(cfg.VAL_FRACTION * n)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

    ds_train = ShardDataset(cfg.SHARDS_DIR, train_idx, cache_size=2)
    ds_val = ShardDataset(cfg.SHARDS_DIR, val_idx, cache_size=2)

    batch_sampler = BalancedBatchSampler(
        ds_train,
        batch_size=cfg.BATCH_SIZE,
        seed=cfg.SEED,
    )

    dl_train = torch.utils.data.DataLoader(
        ds_train,
        batch_sampler=batch_sampler,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_me_fusion,
        pin_memory=True,
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )

    dl_val = torch.utils.data.DataLoader(
        ds_val,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_me_fusion,
        pin_memory=True,
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )

    planes = ("u", "v", "w")
    backbone = make_backbone(cfg.BACKBONE, in_ch=2, embed_dim=cfg.EMBED_DIM).to(device)
    model = MultiViewSetClassifier(backbone=backbone, embed_dim=cfg.EMBED_DIM, plane_names=planes).to(device)
    opt = torch.optim.SGD(
        model.parameters(),
        lr=cfg.LR0,
        momentum=cfg.MOMENTUM,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    loss_fn = nn.BCEWithLogitsLoss()  # unweighted

    it = iter(dl_train)

    for step in range(1, cfg.MAX_STEPS + 1):
        model.train()
        coords_by_plane, feats_by_plane, y, available_mask = next(it)
        y = y.to(device, non_blocking=True)

        # IMPORTANT: keep ME coordinates on CPU; move only features to GPU.
        inputs: Dict[str, ME.SparseTensor] = {}
        for name in planes:
            feats = feats_by_plane[name].to(device, non_blocking=True)
            coords = coords_by_plane[name]  # CPU int32
            inputs[name] = ME.SparseTensor(
                features=feats,
                coordinates=coords,
                device=device,
            )
        logits = model(inputs, available_mask=available_mask.to(device, non_blocking=True)).squeeze(1)
        # Guard against silent broadcasting (a common cause of "stuck at 0.6931/0.5")
        if logits.shape != y.shape:
            raise RuntimeError(
                f"logits shape {tuple(logits.shape)} != y shape {tuple(y.shape)}; "
                "this will broadcast in BCEWithLogitsLoss and can produce zero gradients with balanced batches. "
                "Check ME batching / coordinates batch index."
            )
        loss = loss_fn(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # Poly LR schedule
        lr = poly_lr(step, cfg.MAX_STEPS, cfg.LR0, cfg.POLY_POWER)
        for pg in opt.param_groups:
            pg["lr"] = lr

        if step % 200 == 0:
            with torch.no_grad():
                p = torch.sigmoid(logits)
                acc = ((p > 0.5) == (y > 0.5)).float().mean().item()
            print(f"step {step:7d}  loss {loss.item():.4f}  acc {acc:.3f}  lr {lr:.3e}")

        if step % cfg.VAL_EVERY == 0:
            model.eval()
            tot = 0.0
            cnt = 0
            with torch.no_grad():
                for bi, (coords_by_plane, feats_by_plane, y, available_mask) in enumerate(dl_val):
                    if bi >= cfg.VAL_BATCHES:
                        break
                    y = y.to(device, non_blocking=True)
                    inputs: Dict[str, ME.SparseTensor] = {}
                    for name in planes:
                        feats = feats_by_plane[name].to(device, non_blocking=True)
                        coords = coords_by_plane[name]  # CPU int32
                        inputs[name] = ME.SparseTensor(
                            features=feats,
                            coordinates=coords,
                            device=device,
                        )
                    logits = model(inputs, available_mask=available_mask.to(device, non_blocking=True)).squeeze(1)
                    if logits.shape != y.shape:
                        raise RuntimeError(
                            f"[val] logits shape {tuple(logits.shape)} != y shape {tuple(y.shape)}; "
                            "check ME batching / coordinates."
                        )
                    tot += loss_fn(logits, y).item()
                    cnt += 1
            print(f"[val] step {step:7d}  loss {tot/max(cnt,1):.4f}")
