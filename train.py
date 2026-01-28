#!/usr/bin/env python3
"""
Train MinkUNetClassifier from a ROOT TTree with balanced (50/50) signal/background batches.

Input ROOT branches expected (defaults; override via CLI):
  - is_signal     (bool / int)         : event label (1=signal, 0=background)
  - detector_u    (vector<float>)      : flattened U plane image, length = H*W
  - detector_v    (vector<float>)      : flattened V plane image, length = H*W
  - detector_w    (vector<float>)      : flattened W plane image, length = H*W
  - w_nominal     (double / float)     : event weight (used for *within-class* weighted sampling)

Key behavior:
  - Each mini-batch is split equally: batch_size/2 signal + batch_size/2 background.
  - Within each class, events are sampled with probability ∝ |w_nominal| (replacement sampling).
  - Loss: Binary cross-entropy on logits (BCEWithLogitsLoss).

Dependencies:
  - torch, MinkowskiEngine
  - uproot, awkward, numpy

Example:
  python train_minkunet.py \
    --input train.root --tree Events \
    --height 768 --width 768 \
    --batch-size 8 --epochs 20 --lr 1e-3 \
    --threshold 0.0 --device cuda \
    --out checkpoint.pt
"""

import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import MinkowskiEngine as ME

# ---------- Model (as provided) ----------

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dimension):
        super().__init__()
        self.conv1 = ME.MinkowskiConvolution(
            in_channels, out_channels, kernel_size=3, dimension=dimension
        )
        self.bn1 = ME.MinkowskiBatchNorm(out_channels)
        self.conv2 = ME.MinkowskiConvolution(
            out_channels, out_channels, kernel_size=3, dimension=dimension
        )
        self.bn2 = ME.MinkowskiBatchNorm(out_channels)
        self.relu = ME.MinkowskiReLU(inplace=True)
        if in_channels != out_channels:
            self.shortcut = ME.MinkowskiConvolution(
                in_channels, out_channels, kernel_size=1, dimension=dimension
            )
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
        F = x.F
        scale = self.log_scale.exp()
        F = (F - self.shift) * scale
        return ME.SparseTensor(
            features=F,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )


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
        assert dimension == 3, "This model is configured for 3D coordinates (view,y,x)"

        self.input_norm = InputNorm(in_channels)

        self.conv0 = ME.MinkowskiConvolution(
            in_channels, base_filters, kernel_size=3, dimension=dimension
        )
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
        self.bn_relu = nn.Sequential(
            ME.MinkowskiBatchNorm(base_filters),
            ME.MinkowskiReLU(inplace=True),
        )
        self.global_pool = ME.MinkowskiGlobalPooling()
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


def build_model():
    return MinkUNetClassifier()

# ---------- ROOT Dataset / Sampler ----------

@dataclass
class BranchNames:
    is_signal: str = "is_signal"
    u: str = "detector_u"
    v: str = "detector_v"
    w: str = "detector_w"
    w_nominal: str = "w_nominal"


class RootSparseImageDataset(torch.utils.data.Dataset):
    """
    Stores awkward arrays for U,V,W; returns sparse coords/features for MinkowskiEngine.
    Coordinates are (view, y, x) with view in {0,1,2}.
    """
    def __init__(
        self,
        u_arr,
        v_arr,
        w_arr,
        labels_np: np.ndarray,
        weights_np: np.ndarray,
        indices: np.ndarray,
        height: int,
        width: int,
        threshold: float = 0.0,
        eps_weight: float = 1e-12,
    ):
        super().__init__()
        self.u_arr = u_arr
        self.v_arr = v_arr
        self.w_arr = w_arr

        self.labels_all = labels_np.astype(np.int64)
        self.weights_all = weights_np.astype(np.float64)

        self.indices = indices.astype(np.int64)
        self.H = int(height)
        self.W = int(width)
        self.threshold = float(threshold)
        self.eps_weight = float(eps_weight)

        if self.H <= 0 or self.W <= 0:
            raise ValueError(f"Invalid image shape H={self.H}, W={self.W}")

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def _plane_to_sparse(self, flat: np.ndarray, view_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        flat: shape (H*W,) float32/float64
        Returns:
          coords: (N,3) int32 (view,y,x)
          feats : (N,1) float32
        """
        if flat.ndim != 1:
            flat = flat.reshape(-1)
        if flat.size != self.H * self.W:
            raise ValueError(
                f"Plane length {flat.size} != H*W ({self.H}*{self.W}={self.H*self.W}). "
                "Pass correct --height/--width or fix input."
            )

        thr = self.threshold
        if thr <= 0:
            idx = np.flatnonzero(flat != 0)
        else:
            idx = np.flatnonzero(np.abs(flat) > thr)

        if idx.size == 0:
            return np.zeros((0, 3), dtype=np.int32), np.zeros((0, 1), dtype=np.float32)

        y = (idx // self.W).astype(np.int32)
        x = (idx % self.W).astype(np.int32)
        v = np.full_like(y, int(view_idx), dtype=np.int32)

        coords = np.stack([v, y, x], axis=1).astype(np.int32)  # (N,3)
        feats = flat[idx].astype(np.float32).reshape(-1, 1)    # (N,1)
        return coords, feats

    def __getitem__(self, i: int):
        entry = int(self.indices[i])

        # awkward -> numpy
        u = np.asarray(self.u_arr[entry], dtype=np.float32)
        v = np.asarray(self.v_arr[entry], dtype=np.float32)
        w = np.asarray(self.w_arr[entry], dtype=np.float32)

        cu, fu = self._plane_to_sparse(u, 0)
        cv, fv = self._plane_to_sparse(v, 1)
        cw, fw = self._plane_to_sparse(w, 2)

        coords = np.concatenate([cu, cv, cw], axis=0)
        feats = np.concatenate([fu, fv, fw], axis=0)

        # Handle completely empty event: MinkowskiEngine generally wants >=1 site.
        if coords.shape[0] == 0:
            coords = np.array([[0, 0, 0]], dtype=np.int32)
            feats = np.array([[0.0]], dtype=np.float32)

        y = float(self.labels_all[entry])
        w_nom = float(self.weights_all[entry])
        if not np.isfinite(w_nom):
            w_nom = 0.0

        # For sampling, we will use |w_nominal| clipped > 0.
        w_samp = max(abs(w_nom), self.eps_weight)

        # Return CPU tensors / numpy for ME.utils.sparse_collate
        coords_t = torch.from_numpy(coords)      # int32
        feats_t = torch.from_numpy(feats)        # float32
        return coords_t, feats_t, y, w_samp


class BalancedSignalBackgroundBatchSampler(torch.utils.data.Sampler[List[int]]):
    """
    Batch sampler that yields batches with:
      - batch_size/2 signal indices
      - batch_size/2 background indices
    Within each class, samples with replacement using per-sample weights.

    labels: array-like of shape (N,) with 0/1
    samp_weights: array-like of shape (N,) positive weights used *within each class*
    """
    def __init__(
        self,
        labels: np.ndarray,
        samp_weights: np.ndarray,
        batch_size: int,
        steps_per_epoch: Optional[int] = None,
        seed: int = 12345,
        shuffle_within_batch: bool = True,
    ):
        super().__init__()
        if batch_size % 2 != 0:
            raise ValueError("batch_size must be even to split equally between signal/background.")
        self.batch_size = int(batch_size)
        self.half = self.batch_size // 2
        self.shuffle_within_batch = bool(shuffle_within_batch)

        labels = np.asarray(labels).astype(np.int64)
        w = np.asarray(samp_weights).astype(np.float64)

        self.sig_idx = np.where(labels == 1)[0]
        self.bkg_idx = np.where(labels == 0)[0]
        if self.sig_idx.size == 0 or self.bkg_idx.size == 0:
            raise ValueError(
                f"Need both classes in training set. Found n_sig={self.sig_idx.size}, n_bkg={self.bkg_idx.size}."
            )

        # Normalize weights within each class to probabilities
        ws = w[self.sig_idx].copy()
        wb = w[self.bkg_idx].copy()

        # Ensure positivity
        ws[~np.isfinite(ws)] = 0.0
        wb[~np.isfinite(wb)] = 0.0
        ws = np.clip(ws, 1e-12, None)
        wb = np.clip(wb, 1e-12, None)

        self.ps = ws / ws.sum()
        self.pb = wb / wb.sum()

        # Define epoch length (number of batches)
        if steps_per_epoch is None:
            # Roughly “one pass” over the larger class in terms of examples-per-epoch.
            steps_per_epoch = int(math.ceil(max(self.sig_idx.size, self.bkg_idx.size) / self.half))
        self.steps_per_epoch = int(steps_per_epoch)

        self.seed = int(seed)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)

        for _ in range(self.steps_per_epoch):
            sig = rng.choice(self.sig_idx, size=self.half, replace=True, p=self.ps)
            bkg = rng.choice(self.bkg_idx, size=self.half, replace=True, p=self.pb)
            batch = np.concatenate([sig, bkg]).astype(np.int64)
            if self.shuffle_within_batch:
                rng.shuffle(batch)
            yield batch.tolist()

# ---------- Collate + Train/Eval ----------

def make_collate_fn(device: torch.device):
    def collate(batch):
        # batch: list of (coords_t, feats_t, y, w_samp)
        coords_list = [b[0].int() for b in batch]   # each (Ni,3)
        feats_list = [b[1].float() for b in batch]  # each (Ni,1)
        y = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device).view(-1, 1)
        w_samp = torch.tensor([b[3] for b in batch], dtype=torch.float32, device=device).view(-1, 1)

        # MinkowskiEngine sparse collate adds batch index column
        try:
            coords, feats = ME.utils.sparse_collate(coords_list, feats_list)
        except AttributeError:
            # Fallback if sparse_collate not present in your ME version
            coords = ME.utils.batched_coordinates(coords_list)
            feats = torch.cat(feats_list, dim=0)

        x = ME.SparseTensor(feats, coords, device=device)
        return x, y, w_samp
    return collate


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for x, y, _w in loader:
        out = model(x)
        logits = out.F  # (B,1)
        loss = criterion(logits, y).mean()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


def train_one_epoch(model: nn.Module, loader, optimizer, criterion, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch_idx, (x, y, _w) in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        logits = out.F  # (B,1)
        loss = criterion(logits, y).mean()
        loss.backward()
        optimizer.step()

        print(f"  batch {batch_idx:04d} | loss={loss.item():.6f}")
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


# ---------- Main ----------

def stratified_split(labels: np.ndarray, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels).astype(np.int64)
    rng = np.random.default_rng(seed)

    all_idx = np.arange(labels.shape[0], dtype=np.int64)
    sig = all_idx[labels == 1]
    bkg = all_idx[labels == 0]

    rng.shuffle(sig)
    rng.shuffle(bkg)

    n_sig_val = int(round(sig.size * val_frac))
    n_bkg_val = int(round(bkg.size * val_frac))

    val_idx = np.concatenate([sig[:n_sig_val], bkg[:n_bkg_val]])
    train_idx = np.concatenate([sig[n_sig_val:], bkg[n_bkg_val:]])

    rng.shuffle(val_idx)
    rng.shuffle(train_idx)
    return train_idx, val_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input ROOT file")
    parser.add_argument("--tree", required=True, help="TTree name")
    parser.add_argument("--height", type=int, required=True, help="Image height H (drift-time bins)")
    parser.add_argument("--width", type=int, required=True, help="Image width W (wire bins)")
    parser.add_argument("--threshold", type=float, default=0.0, help="Active-site threshold (abs(value) > thr).")
    parser.add_argument("--batch-size", type=int, default=8, help="Must be even (half signal, half background).")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--steps-per-epoch", type=int, default=None, help="Override number of balanced batches per epoch.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out", default="checkpoint.pt")

    # Branch overrides
    parser.add_argument("--branch-is-signal", default="is_signal")
    parser.add_argument("--branch-u", default="detector_u")
    parser.add_argument("--branch-v", default="detector_v")
    parser.add_argument("--branch-w", default="detector_w")
    parser.add_argument("--branch-weight", default="w_nominal")

    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)

    # Load ROOT
    import uproot
    import awkward as ak

    branches = BranchNames(
        is_signal=args.branch_is_signal,
        u=args.branch_u,
        v=args.branch_v,
        w=args.branch_w,
        w_nominal=args.branch_weight,
    )

    with uproot.open(args.input) as f:
        tree = f[args.tree]
        arr = tree.arrays(
            [branches.is_signal, branches.u, branches.v, branches.w, branches.w_nominal],
            library="ak",
        )

    labels = ak.to_numpy(arr[branches.is_signal]).astype(np.int64)
    weights = ak.to_numpy(arr[branches.w_nominal]).astype(np.float64)

    u_arr = arr[branches.u]
    v_arr = arr[branches.v]
    w_arr = arr[branches.w]

    if labels.ndim != 1:
        labels = labels.reshape(-1)

    # Train/val split (stratified)
    train_idx_global, val_idx_global = stratified_split(labels, args.val_frac, args.seed)

    # Datasets
    train_ds = RootSparseImageDataset(
        u_arr=u_arr,
        v_arr=v_arr,
        w_arr=w_arr,
        labels_np=labels,
        weights_np=weights,
        indices=train_idx_global,
        height=args.height,
        width=args.width,
        threshold=args.threshold,
    )
    val_ds = RootSparseImageDataset(
        u_arr=u_arr,
        v_arr=v_arr,
        w_arr=w_arr,
        labels_np=labels,
        weights_np=weights,
        indices=val_idx_global,
        height=args.height,
        width=args.width,
        threshold=args.threshold,
    )

    # Build sampler using dataset-local indexing
    train_labels_local = labels[train_idx_global]
    train_w_local = np.clip(np.abs(weights[train_idx_global]), 1e-12, None)

    batch_sampler = BalancedSignalBackgroundBatchSampler(
        labels=train_labels_local,
        samp_weights=train_w_local,   # weighted sampling within each class
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        seed=args.seed,
    )

    collate_fn = make_collate_fn(device)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    # Model / optimizer / loss
    model = build_model().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    # Train
    best_val = float("inf")
    for epoch in range(args.epochs):
        # advance sampler epoch seed
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1:03d}/{args.epochs:03d} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        )

        # Save best checkpoint
        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "args": vars(args),
            }
            torch.save(ckpt, args.out)
            print(f"  -> saved: {args.out} (best val_loss so far)")

    print(f"Done. Best val_loss={best_val:.6f} (checkpoint: {args.out})")


if __name__ == "__main__":
    main()
