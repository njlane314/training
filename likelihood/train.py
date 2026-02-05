# train.py
import numpy as np
import torch
import torch.nn as nn
import math
from pathlib import Path
from typing import Dict, Optional

import MinkowskiEngine as ME

from . import config as cfg
from .dataset import BalancedBatchSampler, ShardDataset, collate_me_fusion
from .fusion import MultiViewSetClassifier
from .model import make_backbone

def poly_lr(step, max_steps, lr0, power):
    t = min(step / max_steps, 1.0)
    return lr0 * (1.0 - t) ** power

def _capture_random_state():
    state = {
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _checkpoint_path_for_step(base_path: Path, step: int) -> Path:
    stem = base_path.stem
    suffix = base_path.suffix
    return base_path.with_name(f"{stem}_step{step:07d}{suffix}")


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

    val_batch_sampler = BalancedBatchSampler(ds_val, batch_size=cfg.BATCH_SIZE, seed=cfg.SEED + 999)

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
        batch_sampler=val_batch_sampler,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_me_fusion,
        pin_memory=True,
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )

    # Validation can dominate wall time (extra forward pass + shard I/O) if run every step.
    # Run it periodically instead, and optionally cache a few fixed val batches in RAM.
    #
    # Knobs (all optional; read via getattr so config.py doesn't have to define them):
    #   VAL_EVERY         : validate every N steps (0 disables validation). Default: 200
    #   VAL_NUM_BATCHES   : average validation loss over N batches. Default: 1
    #   VAL_CACHE_BATCHES : prefetch N validation batches at startup (RAM) to avoid shard I/O later. Default: 0
    val_every = int(getattr(cfg, "VAL_EVERY", 200))
    val_num_batches = int(getattr(cfg, "VAL_NUM_BATCHES", 1))
    val_cache_batches = int(getattr(cfg, "VAL_CACHE_BATCHES", 0))
    if val_every < 0:
        raise ValueError("VAL_EVERY must be >= 0")
    if val_num_batches <= 0:
        raise ValueError("VAL_NUM_BATCHES must be >= 1")
    if val_cache_batches < 0:
        raise ValueError("VAL_CACHE_BATCHES must be >= 0")

    cached_val_batches = []
    val_it = iter(dl_val)
    if val_every > 0 and val_cache_batches > 0:
        for _ in range(val_cache_batches):
            try:
                cached_val_batches.append(next(val_it))
            except StopIteration:
                val_it = iter(dl_val)
                cached_val_batches.append(next(val_it))
        print(
            f"[val] cached {len(cached_val_batches)} val batches in RAM "
            f"(VAL_CACHE_BATCHES={val_cache_batches})"
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
    log_path = Path(getattr(cfg, "LOSS_LOG_PATH", "loss.tsv"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Line-buffered logging can be slow on network filesystems. Use a larger buffer and flush occasionally.
    log_flush_every = int(getattr(cfg, "LOG_FLUSH_EVERY", 50))
    log_f = log_path.open("w", buffering=64 * 1024)
    log_f.write("#step\tis_val\tloss\n")

    ckpt_base_path = Path(cfg.CHECKPOINT_PATH)
    ckpt_base_path.parent.mkdir(parents=True, exist_ok=True)

    initial_random_state = _capture_random_state()
    if cfg.CHECKPOINT_EVERY > 0:
        ckpt_path = _checkpoint_path_for_step(ckpt_base_path, step=0)
        torch.save(
            {
                "step": 0,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "initial_random_state": initial_random_state,
                "random_state": initial_random_state,
            },
            ckpt_path,
        )
        print(f"[ckpt] saved initial state -> {ckpt_path}")

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
        log_f.write(f"{step}\t0\t{loss.item():.8g}\n")

        opt.zero_grad(set_to_none=True)
        loss.backward()

        if step % 200 == 0:
            with torch.no_grad():
                print("logits mean/std:", logits.mean().item(), logits.std().item())

            gn = 0.0
            nz = 0
            for p in model.parameters():
                if p.grad is not None:
                    g = p.grad.detach()
                    gn += g.norm().item() ** 2
                    nz += int(g.abs().sum().item() > 0)
            print("grad L2:", (gn ** 0.5), "nonzero_grad_params:", nz)

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

        do_val = (val_every > 0) and (step % val_every == 0 or step == cfg.MAX_STEPS)
        if do_val:
            model.eval()
            with torch.no_grad():
                # If we cached batches, use them (fixed probe set, no shard I/O).
                # Otherwise, draw val_num_batches fresh batches and average.
                if cached_val_batches:
                    batches = cached_val_batches
                else:
                    batches = []
                    for _ in range(val_num_batches):
                        try:
                            batches.append(next(val_it))
                        except StopIteration:
                            val_it = iter(dl_val)
                            batches.append(next(val_it))

                vloss = 0.0
                for coords_by_plane, feats_by_plane, yv, available_mask in batches:
                    yv = yv.to(device, non_blocking=True)
                    inputs: Dict[str, ME.SparseTensor] = {}
                    for name in planes:
                        feats = feats_by_plane[name].to(device, non_blocking=True)
                        coords = coords_by_plane[name]  # CPU int32
                        inputs[name] = ME.SparseTensor(
                            features=feats,
                            coordinates=coords,
                            device=device,
                        )
                    vlogits = model(
                        inputs, available_mask=available_mask.to(device, non_blocking=True)
                    ).squeeze(1)
                    if vlogits.shape != yv.shape:
                        raise RuntimeError(
                            f"[val] logits shape {tuple(vlogits.shape)} != y shape {tuple(yv.shape)}; "
                            "check ME batching / coordinates."
                        )
                    vloss += loss_fn(vlogits, yv).item()
                val_loss = vloss / float(len(batches))

            if step % 200 == 0:
                print(f"[val] step {step:7d}  loss {val_loss:.4f}")
            log_f.write(f"{step}\t1\t{val_loss:.8g}\n")

        if log_flush_every > 0 and step % log_flush_every == 0:
            log_f.flush()

        if cfg.CHECKPOINT_EVERY > 0 and step % cfg.CHECKPOINT_EVERY == 0:
            ckpt_path = _checkpoint_path_for_step(ckpt_base_path, step=step)
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "initial_random_state": initial_random_state,
                    "random_state": _capture_random_state(),
                },
                ckpt_path,
            )
            print(f"[ckpt] saved step {step:7d} -> {ckpt_path}")

    log_f.close()
