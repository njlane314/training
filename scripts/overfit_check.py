#!/usr/bin/env python3
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import MinkowskiEngine as ME

from likelihood import config as cfg
from likelihood.data import ShardDataset, collate
from likelihood.model import MinkUNetClassifier


def main():
    """
    @brief Run a quick overfit check on a small subset.
    """
    meta = torch.load(os.path.join(cfg.SHARDS_DIR, "index.pt"), map_location="cpu")
    labels = np.asarray(meta["labels"], dtype=np.uint8)
    sig = np.where(labels == 1)[0]
    bkg = np.where(labels == 0)[0]

    n = 64
    if sig.size < n or bkg.size < n:
        raise SystemExit("not enough events")

    idx = np.concatenate([sig[:n], bkg[:n]]).astype(np.int64, copy=False)
    rng = np.random.default_rng(cfg.SEED)
    rng.shuffle(idx)

    ds = ShardDataset(cfg.SHARDS_DIR, idx)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, num_workers=0, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MinkUNetClassifier(in_channels=4, base=16, strides=3, dropout=0.0).to(device)
    opt = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    loss_fn = nn.BCEWithLogitsLoss()

    ema = None
    for step in range(1, 801):
        for coords, feats, y in loader:
            x = ME.SparseTensor(feats, coords, device=device)
            y = y.to(device, non_blocking=True).float().view(-1)
            opt.zero_grad(set_to_none=True)
            logits = model(x).view(-1)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            l = float(loss.item())
            ema = l if ema is None else (0.98 * ema + 0.02 * l)
            acc = (logits > 0).eq(y > 0.5).float().mean().item()
            print(f"{step:04d} loss={l:.6f} ema={ema:.6f} acc={acc:.3f}", flush=True)
            break
        if ema is not None and ema < 0.15 and acc > 0.95:
            break

    model.eval()
    tot = 0
    cor = 0
    with torch.no_grad():
        for coords, feats, y in loader:
            x = ME.SparseTensor(feats, coords, device=device)
            y = y.to(device, non_blocking=True).float().view(-1)
            p = (model(x).view(-1) > 0).to(dtype=torch.float32)
            cor += int(p.eq(y > 0.5).sum().item())
            tot += int(y.numel())
    acc = cor / max(tot, 1)
    print(f"final_acc={acc:.4f}", flush=True)
    if acc < 0.90:
        sys.exit(1)


if __name__ == "__main__":
    main()
