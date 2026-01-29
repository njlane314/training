import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

import MinkowskiEngine as ME

from . import config as cfg
from .data import BalancedBatchSampler, ShardDataset, collate
from .model import MinkUNetClassifier


def stratified_split(labels, frac, seed):
    """
    @brief Split indices into stratified train/validation sets.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(labels.shape[0], dtype=np.int64)
    sig = idx[labels == 1]
    bkg = idx[labels == 0]
    rng.shuffle(sig)
    rng.shuffle(bkg)
    ns = int(round(sig.size * frac))
    nb = int(round(bkg.size * frac))
    val = np.concatenate([sig[:ns], bkg[:nb]])
    trn = np.concatenate([sig[ns:], bkg[nb:]])
    rng.shuffle(val)
    rng.shuffle(trn)
    return trn, val


def fixed_probe_indices(labels, batch, seed):
    """
    @brief Select a fixed probe batch for validation monitoring.
    """
    rng = np.random.default_rng(seed)
    h = batch // 2
    sig = np.where(labels == 1)[0]
    bkg = np.where(labels == 0)[0]
    if sig.size < h or bkg.size < h:
        raise ValueError("not enough events for probe batch")
    s = rng.choice(sig, size=h, replace=False)
    b = rng.choice(bkg, size=h, replace=False)
    p = np.concatenate([s, b]).astype(np.int64, copy=False)
    rng.shuffle(p)
    return p


def main():
    """
    @brief Train a MinkowskiEngine UNet classifier end-to-end.
    """
    try:
        mp.set_start_method("forkserver", force=True)
    except RuntimeError:
        pass

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta = torch.load(os.path.join(cfg.SHARDS_DIR, "index.pt"), map_location="cpu")
    labels_all = np.asarray(meta["labels"], dtype=np.uint8)

    trn_idx, val_idx = stratified_split(labels_all, cfg.VAL_FRAC, cfg.SEED)
    trn_idx = np.sort(trn_idx)
    val_idx = np.sort(val_idx)

    trn_ds = ShardDataset(cfg.SHARDS_DIR, trn_idx)
    val_ds = ShardDataset(cfg.SHARDS_DIR, val_idx)

    trn_bs = BalancedBatchSampler(trn_ds.labels, trn_ds.shard_ids, trn_ds.local_ids, cfg.BATCH, cfg.SEED)
    trn_loader = torch.utils.data.DataLoader(
        trn_ds,
        batch_sampler=trn_bs,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.NUM_WORKERS > 0),
        multiprocessing_context="forkserver" if cfg.NUM_WORKERS > 0 else None,
        prefetch_factor=2 if cfg.NUM_WORKERS > 0 else None,
        timeout=120 if cfg.NUM_WORKERS > 0 else 0,
        collate_fn=collate,
    )

    probe_local = fixed_probe_indices(val_ds.labels, cfg.BATCH, cfg.SEED + 999)
    vb = val_ds.__getitems__(probe_local)
    vcoords, vfeats, vy = collate(vb)
    vprobe_x = ME.SparseTensor(vfeats, vcoords, device=device)
    vprobe_y = vy.to(device, non_blocking=True)

    model = MinkUNetClassifier(in_channels=4, base=cfg.BASE_FILTERS, strides=cfg.NUM_STRIDES, dropout=cfg.DROPOUT).to(
        device
    )
    print("\n--- Model Architecture (MinkUNet Classifier) ---")
    print(model)
    print("--------------------------------------------------")
    opt = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    steps_per_epoch = len(trn_loader)
    sched = optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=cfg.LR,
        epochs=cfg.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.05,
        div_factor=10.0,
        final_div_factor=100.0,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    t0 = time.time()
    c, f, y = next(iter(trn_loader))
    t1 = time.time()
    print(f"warmup {t1-t0:.2f}s nnz={int(f.shape[0])} y_mean={float(y.mean()):.3f}", flush=True)

    best = float("inf")
    ema_t = None
    ema_v = None
    step0 = 0

    for epoch in range(cfg.EPOCHS):
        trn_bs.set_epoch(epoch)
        model.train()
        for i, (coords, feats, y) in enumerate(trn_loader):
            x = ME.SparseTensor(feats, coords, device=device)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            tloss = loss_fn(logits, y)
            tloss.backward()
            if cfg.GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            opt.step()
            sched.step()

            model.eval()
            with torch.no_grad():
                vlogits = model(vprobe_x)
                vloss = loss_fn(vlogits, vprobe_y)
            model.train()

            tl = float(tloss.item())
            vl = float(vloss.item())
            ema_t = tl if ema_t is None else (cfg.EMA * ema_t + (1.0 - cfg.EMA) * tl)
            ema_v = vl if ema_v is None else (cfg.EMA * ema_v + (1.0 - cfg.EMA) * vl)

            tacc = (logits > 0).eq(y > 0.5).float().mean().item()
            vacc = (vlogits > 0).eq(vprobe_y > 0.5).float().mean().item()
            lr = opt.param_groups[0]["lr"]
            step = step0 + i + 1
            print(
                f"{epoch+1:03d} {step:07d} lr={lr:.2e} train={tl:.6f} val={vl:.6f} "
                f"ema={ema_t:.6f}/{ema_v:.6f} acc={tacc:.3f} vacc={vacc:.3f}",
                flush=True,
            )

        step0 += len(trn_loader)
        if ema_v is not None and ema_v < best:
            best = float(ema_v)
            torch.save(
                {
                    "epoch": int(epoch + 1),
                    "step": int(step0),
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "best_ema_val": best,
                    "cfg": {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()},
                },
                cfg.OUT,
            )

    torch.save({"epoch": int(cfg.EPOCHS), "step": int(step0), "model": model.state_dict()}, cfg.OUT)
