#!/usr/bin/env python3
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import uproot

import MinkowskiEngine as ME

ROOT = os.environ.get("ROOT_FILE", "/gluster/data/dune/niclane/events.root")
TREE = "events"
OUT = os.environ.get("OUT", "checkpoint.pt")

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dimension):
        super().__init__()
        self.conv1 = ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3, dimension=dimension)
        self.bn1 = ME.MinkowskiBatchNorm(out_channels)
        self.conv2 = ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=3, dimension=dimension)
        self.bn2 = ME.MinkowskiBatchNorm(out_channels)
        self.relu = ME.MinkowskiReLU(inplace=True)
        if in_channels != out_channels:
            self.shortcut = ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=1, dimension=dimension)
        else:
            self.shortcut = None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        return self.relu(out + identity)


class InputNorm(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(num_channels))
        self.log_scale = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        F = (x.F - self.shift) * self.log_scale.exp()
        return ME.SparseTensor(features=F, coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager)


class MinkUNetClassifier(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        dimension=3,
        base_filters=16,
        num_strides=3,
    ):
        super().__init__()
        assert dimension == 3

        self.input_norm = InputNorm(in_channels)

        self.conv0 = ME.MinkowskiConvolution(in_channels, base_filters, kernel_size=3, dimension=dimension)
        self.encoder = nn.ModuleList()
        ch = base_filters
        for _ in range(num_strides):
            self.encoder.append(ResidualBlock(ch, ch * 2, dimension))
            self.encoder.append(
                ME.MinkowskiConvolution(
                    ch * 2,
                    ch * 2,
                    kernel_size=(1, 2, 2),
                    stride=(1, 2, 2),
                    dimension=dimension,
                )
            )
            ch *= 2
        self.bottleneck = ResidualBlock(ch, ch, dimension)
        self.decoder = nn.ModuleList()
        for i in range(num_strides):
            up = ch // 2
            self.decoder.append(
                ME.MinkowskiConvolutionTranspose(
                    ch,
                    up,
                    kernel_size=(1, 2, 2),
                    stride=(1, 2, 2),
                    dimension=dimension,
                )
            )
            skip_ch = base_filters * (2 ** (num_strides - i))
            self.decoder.append(ResidualBlock(up + skip_ch, up, dimension))
            ch = up
        self.bn_relu = nn.Sequential(ME.MinkowskiBatchNorm(base_filters), ME.MinkowskiReLU(inplace=True))
        self.global_pool = ME.MinkowskiGlobalAvgPooling()
        self.linear = ME.MinkowskiLinear(base_filters, out_channels)

    def forward(self, x):
        x = self.input_norm(x)
        x = self.conv0(x)
        skips = []
        for i in range(0, len(self.encoder), 2):
            x = self.encoder[i](x)
            skips.append(x)
            x = self.encoder[i + 1](x)
        x = self.bottleneck(x)
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)
            skip = skips.pop()
            x = ME.cat(x, skip)
            x = self.decoder[i + 1](x)
        x = self.bn_relu(x)
        x = self.global_pool(x)
        x = self.linear(x)
        return x


H = 512
W = 512
THRESH = 0.0
BATCH = 64
EPOCHS = 1
LR = 1e-2
WEIGHT_DECAY = 0.0
VAL_FRAC = 0.1
NUM_WORKERS = 8
SEED = 12345

BR_Y = "is_signal"
BR_U = "detector_image_u"
BR_V = "detector_image_v"
BR_W = "detector_image_w"
BR_WGT = "w_nominal"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def split(labels, frac, seed):
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


def plane_to_sparse(flat, view, H, W, thr):
    if flat.ndim != 1:
        flat = flat.reshape(-1)
    if thr <= 0.0:
        idx = np.flatnonzero(flat)
    else:
        idx = np.flatnonzero(np.abs(flat) > thr)
    if idx.size == 0:
        return None, None
    y, x = np.divmod(idx.astype(np.int64, copy=False), W)
    n = idx.size
    coords = np.empty((n, 3), dtype=np.int32)
    coords[:, 0] = int(view)
    coords[:, 1] = y.astype(np.int32, copy=False)
    coords[:, 2] = x.astype(np.int32, copy=False)
    feats = flat[idx].astype(np.float32, copy=False).reshape(n, 1)
    return coords, feats


class RootDataset(torch.utils.data.Dataset):
    def __init__(self, path, tree, entries, labels, samp_w):
        self.path = path
        self.tree = tree
        self.entries = entries.astype(np.int64, copy=False)
        self.labels = labels.astype(np.float32, copy=False)
        self.samp_w = samp_w.astype(np.float64, copy=False)
        self._t = None
        self._f = None

    def _get(self):
        if self._t is None:
            self._f = uproot.open(self.path)
            self._t = self._f[self.tree]
        return self._t

    def __len__(self):
        return int(self.entries.shape[0])

    def __getitem__(self, i):
        entry = int(self.entries[i])
        t = self._get()
        a = t.arrays([BR_U, BR_V, BR_W], entry_start=entry, entry_stop=entry + 1, library="np")
        u = a[BR_U][0]
        v = a[BR_V][0]
        w = a[BR_W][0]
        cu, fu = plane_to_sparse(u, 0, H, W, THRESH)
        cv, fv = plane_to_sparse(v, 1, H, W, THRESH)
        cw, fw = plane_to_sparse(w, 2, H, W, THRESH)
        coords = []
        feats = []
        if cu is not None:
            coords.append(cu)
            feats.append(fu)
        if cv is not None:
            coords.append(cv)
            feats.append(fv)
        if cw is not None:
            coords.append(cw)
            feats.append(fw)
        if not coords:
            c = np.array([[0, 0, 0]], dtype=np.int32)
            f = np.array([[0.0]], dtype=np.float32)
        else:
            c = np.concatenate(coords, axis=0)
            f = np.concatenate(feats, axis=0)
        return torch.from_numpy(c), torch.from_numpy(f), float(self.labels[i])


class BalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, labels, samp_w, entries, batch_size, seed):
        if batch_size % 2:
            raise ValueError("batch_size must be even")
        self.bs = int(batch_size)
        self.h = self.bs // 2
        self.labels = labels.astype(np.int64, copy=False)
        self.entries = entries.astype(np.int64, copy=False)
        w = np.clip(np.abs(samp_w.astype(np.float64, copy=False)), 1e-12, None)
        self.sig = np.where(self.labels == 1)[0]
        self.bkg = np.where(self.labels == 0)[0]
        if self.sig.size == 0 or self.bkg.size == 0:
            raise ValueError("need both classes")
        ws = w[self.sig]
        wb = w[self.bkg]
        self.ps = ws / ws.sum()
        self.pb = wb / wb.sum()
        self.steps = int(math.ceil(max(self.sig.size, self.bkg.size) / self.h))
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, e):
        self.epoch = int(e)

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.steps):
            s = rng.choice(self.sig, size=self.h, replace=True, p=self.ps)
            b = rng.choice(self.bkg, size=self.h, replace=True, p=self.pb)
            batch = np.concatenate([s, b])
            batch = batch[np.argsort(self.entries[batch])]
            yield batch.tolist()


def collate(batch):
    coords, feats = ME.utils.sparse_collate([b[0] for b in batch], [b[1] for b in batch])
    y = torch.tensor([b[2] for b in batch], dtype=torch.float32).view(-1, 1)
    return coords, feats, y


def main():
    try:
        mp.set_start_method("forkserver", force=True)
    except RuntimeError:
        pass

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with uproot.open(ROOT) as f:
        t = f[TREE]
        labels = t[BR_Y].array(library="np").astype(np.int64).reshape(-1)
        w = t[BR_WGT].array(library="np").astype(np.float64).reshape(-1)

    trn_entries, val_entries = split(labels, VAL_FRAC, SEED)
    trn_labels = labels[trn_entries]
    val_labels = labels[val_entries]
    trn_w = np.clip(np.abs(w[trn_entries]), 1e-12, None)
    val_w = np.clip(np.abs(w[val_entries]), 1e-12, None)

    trn_ds = RootDataset(ROOT, TREE, trn_entries, trn_labels, trn_w)
    val_ds = RootDataset(ROOT, TREE, val_entries, val_labels, val_w)

    trn_bs = BalancedBatchSampler(trn_labels, trn_w, trn_entries, BATCH, SEED)
    val_bs = BalancedBatchSampler(val_labels, val_w, val_entries, BATCH, SEED + 999)

    trn_loader = torch.utils.data.DataLoader(
        trn_ds,
        batch_sampler=trn_bs,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        multiprocessing_context="forkserver" if NUM_WORKERS > 0 else None,
        timeout=120 if NUM_WORKERS > 0 else 0,
        collate_fn=collate,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_sampler=val_bs,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate,
    )

    t0 = time.time()
    c, f, y = next(iter(trn_loader))
    t1 = time.time()
    vc, vf, vy = next(iter(val_loader))
    t2 = time.time()
    print(
        f"warmup train_batch={t1-t0:.2f}s val_batch={t2-t1:.2f}s nnz={int(f.shape[0])} "
        f"y_mean={float(y.mean()):.3f}",
        flush=True,
    )

    model = MinkUNetClassifier().to(device)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()

    best = float("inf")
    step0 = 0
    for epoch in range(EPOCHS):
        trn_bs.set_epoch(epoch)
        val_bs.set_epoch(epoch)
        val_it = iter(val_loader)
        model.train()
        vmean = 0.0
        for i, (coords, feats, y) in enumerate(trn_loader):
            x = ME.SparseTensor(feats, coords, device=device)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x).F
            trn_loss = loss_fn(logits, y)
            trn_loss.backward()
            opt.step()

            try:
                vcoords, vfeats, vy = next(val_it)
            except StopIteration:
                val_it = iter(val_loader)
                vcoords, vfeats, vy = next(val_it)

            model.eval()
            with torch.no_grad():
                vx = ME.SparseTensor(vfeats, vcoords, device=device)
                vy = vy.to(device, non_blocking=True)
                vloss = loss_fn(model(vx).F, vy)
            model.train()

            step = step0 + i + 1
            vmean += (vloss.item() - vmean) / (i + 1)
            print(f"{epoch+1:03d} {step:07d} train={trn_loss.item():.6f} val={vloss.item():.6f}", flush=True)

        step0 += len(trn_loader)
        if vmean < best:
            best = vmean
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "val": best}, OUT)


if __name__ == "__main__":
    main()
