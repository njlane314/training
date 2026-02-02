# train.py
import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME

from . import config as cfg
from .dataset import ShardDataset, InfiniteCorrectedBalancedBatchSampler, collate_me


class ResBlock(nn.Module):
    def __init__(self, cin, cout, D=3, ks=(3, 3, 3)):
        super().__init__()
        self.conv1 = ME.MinkowskiSubmanifoldConvolution(cin, cout, kernel_size=ks, dimension=D, bias=False)
        self.bn1 = ME.MinkowskiBatchNorm(cout)
        self.conv2 = ME.MinkowskiSubmanifoldConvolution(cout, cout, kernel_size=ks, dimension=D, bias=False)
        self.bn2 = ME.MinkowskiBatchNorm(cout)
        self.relu = ME.MinkowskiReLU(inplace=True)

        self.proj = None
        if cin != cout:
            self.proj = nn.Sequential(
                ME.MinkowskiLinear(cin, cout, bias=False),
                ME.MinkowskiBatchNorm(cout),
            )

    def forward(self, x):
        identity = x if self.proj is None else self.proj(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class SparseUResNetEncoderClassifier(nn.Module):
    """
    UResNet-style residual encoder + global pooling head for event-level LLR.
    Input coords: (batch, plane, y, x) with D=3 spatial dims (plane,y,x).
    """

    def __init__(self, in_ch=3, base=32, D=3):
        super().__init__()
        self.D = D

        # Keep it simple: isotropic kernels, downsample only in (y,x)
        self.stem = nn.Sequential(
            ME.MinkowskiSubmanifoldConvolution(in_ch, base, kernel_size=3, dimension=D, bias=False),
            ME.MinkowskiBatchNorm(base),
            ME.MinkowskiReLU(inplace=True),
        )

        self.b0 = ResBlock(base, base, D=D)
        self.down1 = ME.MinkowskiConvolution(base, base * 2, kernel_size=(1, 2, 2), stride=(1, 2, 2), dimension=D, bias=False)
        self.b1 = ResBlock(base * 2, base * 2, D=D)

        self.down2 = ME.MinkowskiConvolution(base * 2, base * 4, kernel_size=(1, 2, 2), stride=(1, 2, 2), dimension=D, bias=False)
        self.b2 = ResBlock(base * 4, base * 4, D=D)

        self.down3 = ME.MinkowskiConvolution(base * 4, base * 8, kernel_size=(1, 2, 2), stride=(1, 2, 2), dimension=D, bias=False)
        self.b3 = ResBlock(base * 8, base * 8, D=D)

        self.pool = ME.MinkowskiGlobalMaxPooling()
        self.head = nn.Linear(base * 8, 1)

    def forward(self, x: ME.SparseTensor):
        x = self.stem(x)
        x = self.b0(x)
        x = self.b1(self.down1(x))
        x = self.b2(self.down2(x))
        x = self.b3(self.down3(x))
        x = self.pool(x)          # one feature vector per batch item
        return self.head(x.F).squeeze(1)  # logits


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
