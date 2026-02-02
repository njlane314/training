# train_llr.py
import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME

from . import config as cfg
from .dataset import ShardDataset, InfiniteCorrectedBalancedBatchSampler, collate_me
from .model import SparseUResNetEncoderClassifier


def poly_lr(step, max_steps, lr0, power):
    t = min(step / max_steps, 1.0)
    return lr0 * (1.0 - t) ** power


def train_llr():
    torch.manual_seed(cfg.LLR_SEED)
    np.random.seed(cfg.LLR_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta = torch.load(f"{cfg.LLR_SHARDS_DIR}/index.pt", map_location="cpu")
    n = int(meta["n_events"])
    rng = np.random.default_rng(cfg.LLR_SEED)
    perm = rng.permutation(n)
    n_val = int(cfg.LLR_VAL_FRACTION * n)

    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    ds_train = ShardDataset(cfg.LLR_SHARDS_DIR, train_idx, cache_size=2)
    ds_val = ShardDataset(cfg.LLR_SHARDS_DIR, val_idx, cache_size=2)

    batch_sampler = InfiniteCorrectedBalancedBatchSampler(
        ds_train,
        batch_size=cfg.LLR_BATCH_SIZE,
        seed=cfg.LLR_SEED,
    )

    dl_train = torch.utils.data.DataLoader(
        ds_train,
        batch_sampler=batch_sampler,
        num_workers=cfg.LLR_NUM_WORKERS,
        collate_fn=collate_me,
        pin_memory=True,
        persistent_workers=(cfg.LLR_NUM_WORKERS > 0),
    )

    dl_val = torch.utils.data.DataLoader(
        ds_val,
        batch_size=cfg.LLR_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.LLR_NUM_WORKERS,
        collate_fn=collate_me,
        pin_memory=True,
        persistent_workers=(cfg.LLR_NUM_WORKERS > 0),
    )

    model = SparseUResNetEncoderClassifier(in_ch=3, base=32, D=3).to(device)
    opt = torch.optim.SGD(
        model.parameters(),
        lr=cfg.LLR_LR0,
        momentum=cfg.LLR_MOMENTUM,
        weight_decay=cfg.LLR_WEIGHT_DECAY,
    )
    loss_fn = nn.BCEWithLogitsLoss()  # unweighted

    it = iter(dl_train)

    for step in range(1, cfg.LLR_MAX_STEPS + 1):
        model.train()
        coords, feats, y = next(it)
        coords = coords.to(device, non_blocking=True)
        feats = feats.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x = ME.SparseTensor(features=feats, coordinates=coords, device=device)
        logits = model(x)
        loss = loss_fn(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # Poly LR schedule
        lr = poly_lr(step, cfg.LLR_MAX_STEPS, cfg.LLR_LR0, cfg.LLR_POLY_POWER)
        for pg in opt.param_groups:
            pg["lr"] = lr

        if step % 200 == 0:
            with torch.no_grad():
                p = torch.sigmoid(logits)
                acc = ((p > 0.5) == (y > 0.5)).float().mean().item()
            print(f"step {step:7d}  loss {loss.item():.4f}  acc {acc:.3f}  lr {lr:.3e}")

        if step % cfg.LLR_VAL_EVERY == 0:
            model.eval()
            tot = 0.0
            cnt = 0
            with torch.no_grad():
                for bi, (coords, feats, y) in enumerate(dl_val):
                    if bi >= cfg.LLR_VAL_BATCHES:
                        break
                    coords = coords.to(device, non_blocking=True)
                    feats = feats.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    x = ME.SparseTensor(features=feats, coordinates=coords, device=device)
                    logits = model(x)
                    tot += loss_fn(logits, y).item()
                    cnt += 1
            print(f"[val] step {step:7d}  loss {tot/max(cnt,1):.4f}")
