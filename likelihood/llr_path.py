#!/usr/bin/env python3
"""
llr_path.py

Train with periodic checkpoints + diagnose whether logits behave like an LLR surrogate.

Diagnostics per checkpoint on a fixed held-out (val) split:
  1) Weighted, class-conditional ROC AUC (pairs drawn with prob ∝ w_nominal within each class).
  2) "LLR in score space": r(z) = log p_s(z) / p_b(z) estimated from weighted histograms,
     compared to the identity line r(z) = z.

Outputs (in --out-dir):
  - metrics.csv
  - llr_consistency.png
  - auc_vs_step.png
  - alpha_beta_mse_vs_step.png

Assumptions:
  - This file lives next to config.py, dataset.py, model.py, fusion.py (same directory),
    OR inside the same package (relative imports fallback).
  - SHARDS_DIR contains index.pt + shard_XXXXX.pt as produced by process.py.

Run:
  python llr_path.py train   --ckpt-dir checkpoints
  python llr_path.py analyze --ckpt-dir checkpoints --out-dir llr_diagnostics
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")  # safe for batch/cluster jobs
import matplotlib.pyplot as plt

import MinkowskiEngine as ME

# Robust imports whether run as script or module
try:
    from . import config as cfg  # type: ignore
    from .dataset import BalancedBatchSampler, ShardDataset, collate_me_fusion  # type: ignore
    from .fusion import MultiViewSetClassifier  # type: ignore
    from .model import make_backbone  # type: ignore
except Exception:
    import config as cfg  # type: ignore
    from dataset import BalancedBatchSampler, ShardDataset, collate_me_fusion  # type: ignore
    from fusion import MultiViewSetClassifier  # type: ignore
    from model import make_backbone  # type: ignore


PLANES = ("u", "v", "w")


def poly_lr(step: int, max_steps: int, lr0: float, power: float) -> float:
    t = min(step / max_steps, 1.0)
    return float(lr0) * (1.0 - t) ** float(power)


def _load_meta(shards_dir: str) -> dict:
    return torch.load(f"{shards_dir}/index.pt", map_location="cpu")


def _compute_splits_from_meta(
    meta: dict,
    *,
    seed: int,
    val_fraction: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match the split logic in your train.py:
      - optionally filter nnz>0 (if present in index.pt)
      - RNG permute
      - take first VAL_FRACTION as val, rest train
    """
    n = int(meta["n_events"])
    rng = np.random.default_rng(int(seed))
    labels_all = np.asarray(meta["labels"], dtype=np.uint8).reshape(-1)
    if labels_all.shape[0] != n:
        raise ValueError(f"index.pt labels len={labels_all.shape[0]} != n_events={n}")

    nnz_all: Optional[np.ndarray] = None
    if "nnz" in meta and meta["nnz"] is not None:
        if isinstance(meta["nnz"], torch.Tensor):
            nnz_all = meta["nnz"].to(dtype=torch.int64).cpu().numpy().reshape(-1)
        else:
            nnz_all = np.asarray(meta["nnz"], dtype=np.int64).reshape(-1)
        if nnz_all.shape[0] != n:
            raise ValueError(f"index.pt nnz len={nnz_all.shape[0]} != n_events={n}")

    if nnz_all is not None:
        good = nnz_all > 0
        idx_all = np.flatnonzero(good)
        if idx_all.size == 0:
            raise ValueError(
                "All events have nnz==0 after sparsification. "
                "Check THRESH / branch names / shard generation (bad_events) in index.pt."
            )

        perm = rng.permutation(idx_all.size)
        idx_perm = idx_all[perm]
        n_val = int(float(val_fraction) * idx_perm.size)
        val_idx = idx_perm[:n_val]
        train_idx = idx_perm[n_val:]

        labs_train = labels_all[train_idx]
        if labs_train.min() == labs_train.max():
            raise ValueError(
                "After filtering nnz>0, the training split contains only one class. "
                "Lower THRESH or inspect index.pt (labels/nnz/bad_events)."
            )
    else:
        perm = rng.permutation(n)
        n_val = int(float(val_fraction) * n)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

    return train_idx.astype(np.int64), val_idx.astype(np.int64)


class ShardDatasetWithWeights(ShardDataset):
    """
    Same as ShardDataset, but returns weight too.
    """

    def __getitem__(self, i: int):
        coords, feats, y = super().__getitem__(i)
        w = float(self.weights[i])
        return coords, feats, y, w


def collate_me_fusion_with_weights(batch, plane_names=PLANES):
    """
    Batch items: (coords, feats, y, w)
    Returns: coords_by_plane, feats_by_plane, y, w, available_mask
    """
    base = [(c, f, y) for (c, f, y, w) in batch]
    coords, feats, y, available_mask = collate_me_fusion(base, plane_names=plane_names)
    w = torch.tensor([w for (c, f, y, w) in batch], dtype=torch.float32)
    return coords, feats, y, w, available_mask


def _build_model_from_cfg(device: torch.device) -> nn.Module:
    backbone = make_backbone(cfg.BACKBONE, in_ch=2, embed_dim=cfg.EMBED_DIM).to(device)
    model = MultiViewSetClassifier(backbone=backbone, embed_dim=cfg.EMBED_DIM, plane_names=PLANES).to(device)
    return model


def _make_loaders(
    shards_dir: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    num_workers: int,
    batch_size: int,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    ds_train = ShardDataset(shards_dir, train_idx, cache_size=2)
    ds_val = ShardDatasetWithWeights(shards_dir, val_idx, cache_size=2)

    batch_sampler = BalancedBatchSampler(
        ds_train,
        batch_size=int(batch_size),
        seed=int(cfg.SEED),
    )

    dl_train = torch.utils.data.DataLoader(
        ds_train,
        batch_sampler=batch_sampler,
        num_workers=int(num_workers),
        collate_fn=collate_me_fusion,
        pin_memory=True,
        persistent_workers=(int(num_workers) > 0),
    )

    dl_val = torch.utils.data.DataLoader(
        ds_val,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collate_me_fusion_with_weights,
        pin_memory=True,
        persistent_workers=(int(num_workers) > 0),
    )

    return dl_train, dl_val


def _step_to_inputs(
    coords_by_plane: Dict[str, torch.Tensor],
    feats_by_plane: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, ME.SparseTensor]:
    inputs: Dict[str, ME.SparseTensor] = {}
    for name in PLANES:
        feats = feats_by_plane[name].to(device, non_blocking=True)
        coords = coords_by_plane[name]  # keep CPU int32
        inputs[name] = ME.SparseTensor(
            features=feats,
            coordinates=coords,
            device=device,
        )
    return inputs


def save_checkpoint(
    path: Path,
    *,
    step: int,
    model: nn.Module,
    opt: Optional[torch.optim.Optimizer],
    val_loss: Optional[float],
) -> None:
    payload = {
        "step": int(step),
        "time_unix": float(time.time()),
        "model": model.state_dict(),
        "opt": (opt.state_dict() if opt is not None else None),
        "val_loss": (float(val_loss) if val_loss is not None else None),
        "cfg": {
            "BACKBONE": str(cfg.BACKBONE),
            "EMBED_DIM": int(cfg.EMBED_DIM),
            "H": int(cfg.H),
            "W": int(cfg.W),
            "THRESH": float(cfg.THRESH),
            "BATCH_SIZE": int(cfg.BATCH_SIZE),
            "SEED": int(cfg.SEED),
        },
    }
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(payload, tmp)
    tmp.replace(path)


def _find_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    ckpts = sorted(ckpt_dir.glob("ckpt_step*.pt"))
    if not ckpts:
        return None
    # Sort by step parsed from filename
    def step_of(p: Path) -> int:
        m = re.search(r"ckpt_step(\d+)\.pt$", p.name)
        return int(m.group(1)) if m else -1
    ckpts = sorted(ckpts, key=step_of)
    return ckpts[-1]


def train_with_checkpoints(
    *,
    ckpt_dir: Path,
    max_steps: int,
    save_every: int,
    resume: bool,
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(cfg.SEED))
    np.random.seed(int(cfg.SEED))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta = _load_meta(cfg.SHARDS_DIR)
    train_idx, val_idx = _compute_splits_from_meta(meta, seed=cfg.SEED, val_fraction=cfg.VAL_FRACTION)

    splits_path = ckpt_dir / "splits.npz"
    if not splits_path.exists():
        np.savez(splits_path, train_idx=train_idx, val_idx=val_idx)

    dl_train, dl_val = _make_loaders(
        cfg.SHARDS_DIR,
        train_idx,
        val_idx,
        num_workers=cfg.NUM_WORKERS,
        batch_size=cfg.BATCH_SIZE,
    )

    model = _build_model_from_cfg(device)
    opt = torch.optim.SGD(
        model.parameters(),
        lr=float(cfg.LR0),
        momentum=float(cfg.MOMENTUM),
        weight_decay=float(cfg.WEIGHT_DECAY),
    )
    loss_fn = nn.BCEWithLogitsLoss()

    start_step = 0
    if resume:
        last = _find_latest_checkpoint(ckpt_dir)
        if last is not None:
            ck = torch.load(last, map_location="cpu")
            model.load_state_dict(ck["model"], strict=True)
            if ck.get("opt") is not None:
                opt.load_state_dict(ck["opt"])
            start_step = int(ck.get("step", 0))
            print(f"[resume] loaded {last} (step={start_step})")

    # Save step-0 checkpoint if starting fresh
    if start_step == 0:
        save_checkpoint(ckpt_dir / f"ckpt_step{0:07d}.pt", step=0, model=model, opt=opt, val_loss=None)

    it = iter(dl_train)

    for step in range(start_step + 1, int(max_steps) + 1):
        model.train()
        coords_by_plane, feats_by_plane, y, available_mask = next(it)
        y = y.to(device, non_blocking=True)

        inputs = _step_to_inputs(coords_by_plane, feats_by_plane, device)
        logits = model(inputs, available_mask=available_mask.to(device, non_blocking=True)).squeeze(1)

        if logits.shape != y.shape:
            raise RuntimeError(
                f"logits shape {tuple(logits.shape)} != y shape {tuple(y.shape)}; "
                "this will broadcast in BCEWithLogitsLoss."
            )

        loss = loss_fn(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        lr = poly_lr(step, int(max_steps), float(cfg.LR0), float(cfg.POLY_POWER))
        for pg in opt.param_groups:
            pg["lr"] = lr

        if step % 200 == 0:
            with torch.no_grad():
                p = torch.sigmoid(logits)
                acc = ((p > 0.5) == (y > 0.5)).float().mean().item()
            print(f"step {step:7d}  loss {loss.item():.4f}  acc {acc:.3f}  lr {lr:.3e}")

        val_loss: Optional[float] = None
        if (step % int(save_every) == 0) or (step == int(max_steps)):
            # quick-ish val loss estimate (same logic as train.py)
            model.eval()
            tot = 0.0
            cnt = 0
            with torch.no_grad():
                for bi, (coords_by_plane, feats_by_plane, yv, wv, available_mask) in enumerate(dl_val):
                    if bi >= int(cfg.VAL_BATCHES):
                        break
                    yv = yv.to(device, non_blocking=True)
                    inputs = _step_to_inputs(coords_by_plane, feats_by_plane, device)
                    lv = model(inputs, available_mask=available_mask.to(device, non_blocking=True)).squeeze(1)
                    if lv.shape != yv.shape:
                        raise RuntimeError(
                            f"[val] logits shape {tuple(lv.shape)} != y shape {tuple(yv.shape)}"
                        )
                    tot += loss_fn(lv, yv).item()
                    cnt += 1
            val_loss = float(tot / max(cnt, 1))
            print(f"[val] step {step:7d}  loss {val_loss:.4f}")

            save_checkpoint(
                ckpt_dir / f"ckpt_step{step:07d}.pt",
                step=step,
                model=model,
                opt=opt,
                val_loss=val_loss,
            )


def _weighted_auc_pairwise(
    z: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
) -> float:
    """
    Weighted AUC with class-conditional sampling:
      draw s ~ w within signal, b ~ w within bkg, compute P(z_s > z_b) + 0.5 P(equal).

    This matches the "balanced priors" view and is consistent with the LLR story.
    """
    y = y.astype(np.uint8, copy=False)
    w = np.clip(w.astype(np.float64, copy=False), 0.0, None)

    z_s = z[y == 1]
    z_b = z[y == 0]
    w_s = w[y == 1]
    w_b = w[y == 0]

    if z_s.size == 0 or z_b.size == 0:
        raise ValueError("need both classes for AUC")

    sw = w_s.sum()
    bw = w_b.sum()
    if sw <= 0 or bw <= 0:
        raise ValueError("weights must sum to >0 within each class for weighted AUC")

    w_s = w_s / sw
    order_b = np.argsort(z_b, kind="mergesort")
    z_b = z_b[order_b]
    w_b = (w_b[order_b] / bw)

    cdf_b = np.cumsum(w_b)  # length n_b
    cdf0 = np.concatenate([[0.0], cdf_b])  # length n_b+1; cdf0[k]=sum_{i<k} w_b[i]

    idx_lt = np.searchsorted(z_b, z_s, side="left")
    idx_le = np.searchsorted(z_b, z_s, side="right")

    p_lt = cdf0[idx_lt]
    p_le = cdf0[idx_le]
    p_eq = p_le - p_lt

    auc = float(np.sum(w_s * (p_lt + 0.5 * p_eq)))
    return auc


@dataclass
class LLRConsistency:
    step: int
    n_val: int
    auc_w: float
    alpha: float
    beta: float
    mse: float
    n_bins_used: int
    z_lo: float
    z_hi: float
    val_loss: Optional[float]


def _llr_consistency_from_scores(
    z: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    *,
    nbins: int,
    q_lo: float,
    q_hi: float,
) -> Tuple[LLRConsistency, np.ndarray, np.ndarray]:
    """
    Returns:
      - metrics (alpha,beta,mse,auc,...)
      - x_centers_used
      - r_used = log ps/pb in score space at those centers
    """
    z = z.astype(np.float64, copy=False)
    y = y.astype(np.uint8, copy=False)
    w = np.clip(w.astype(np.float64, copy=False), 0.0, None)

    if z.size == 0:
        raise ValueError("empty score array")

    z_lo = float(np.quantile(z, float(q_lo)))
    z_hi = float(np.quantile(z, float(q_hi)))
    if not np.isfinite(z_lo) or not np.isfinite(z_hi) or z_hi <= z_lo:
        # fallback: symmetric window around mean
        m = float(np.mean(z))
        s = float(np.std(z) + 1e-6)
        z_lo, z_hi = m - 4.0 * s, m + 4.0 * s

    edges = np.linspace(z_lo, z_hi, int(nbins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    ys = (y == 1)
    yb = (y == 0)

    if ys.sum() == 0 or yb.sum() == 0:
        raise ValueError("need both classes for LLR consistency histograms")

    ws = w[ys]
    wb = w[yb]
    if ws.sum() <= 0 or wb.sum() <= 0:
        raise ValueError("weights must sum to >0 within each class for histograms")

    hs, _ = np.histogram(z[ys], bins=edges, weights=ws)
    hb, _ = np.histogram(z[yb], bins=edges, weights=wb)

    # class-conditional probabilities per bin (bin widths cancel in ratio since widths are equal)
    ps = hs / max(hs.sum(), 1e-300)
    pb = hb / max(hb.sum(), 1e-300)

    mask = (ps > 0) & (pb > 0) & np.isfinite(ps) & np.isfinite(pb)
    x = centers[mask]
    r = (np.log(ps[mask]) - np.log(pb[mask]))

    if x.size < 2:
        # Not enough overlap in score space to fit.
        alpha = float("nan")
        beta = float("nan")
        mse = float("nan")
        n_used = int(x.size)
    else:
        # Linear fit r ≈ alpha*z + beta over overlap region
        alpha, beta = np.polyfit(x, r, deg=1)
        mse = float(np.mean((r - x) ** 2))
        n_used = int(x.size)

    auc_w = _weighted_auc_pairwise(z=z, y=y, w=w)

    # step/val_loss are filled by caller (from checkpoint)
    dummy = LLRConsistency(
        step=-1,
        n_val=int(z.size),
        auc_w=float(auc_w),
        alpha=float(alpha),
        beta=float(beta),
        mse=float(mse),
        n_bins_used=int(n_used),
        z_lo=float(z_lo),
        z_hi=float(z_hi),
        val_loss=None,
    )
    return dummy, x, r


@torch.no_grad()
def _infer_scores_on_val(
    model: nn.Module,
    dl_val: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    max_batches: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    zs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    ws: List[np.ndarray] = []

    model.eval()

    for bi, (coords_by_plane, feats_by_plane, y, w, available_mask) in enumerate(dl_val):
        if max_batches is not None and bi >= int(max_batches):
            break
        y = y.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True)

        inputs = _step_to_inputs(coords_by_plane, feats_by_plane, device)
        logits = model(inputs, available_mask=available_mask.to(device, non_blocking=True)).squeeze(1)

        if logits.shape != y.shape:
            raise RuntimeError(f"[val] logits shape {tuple(logits.shape)} != y shape {tuple(y.shape)}")

        zs.append(logits.detach().cpu().numpy().astype(np.float64, copy=False))
        ys.append(y.detach().cpu().numpy().astype(np.float32, copy=False))
        ws.append(w.detach().cpu().numpy().astype(np.float64, copy=False))

    z = np.concatenate(zs, axis=0) if zs else np.zeros((0,), dtype=np.float64)
    y = np.concatenate(ys, axis=0) if ys else np.zeros((0,), dtype=np.float32)
    w = np.concatenate(ws, axis=0) if ws else np.zeros((0,), dtype=np.float64)
    return z, y, w


def _list_checkpoints(ckpt_dir: Path) -> List[Path]:
    paths = [Path(p) for p in glob.glob(str(ckpt_dir / "ckpt_step*.pt"))]
    if not paths:
        raise FileNotFoundError(f"no checkpoints found in {ckpt_dir} matching ckpt_step*.pt")

    def step_of(p: Path) -> int:
        m = re.search(r"ckpt_step(\d+)\.pt$", p.name)
        if m:
            return int(m.group(1))
        # fallback: load and read step
        ck = torch.load(p, map_location="cpu")
        return int(ck.get("step", -1))

    paths.sort(key=step_of)
    return paths


def analyze_checkpoints(
    *,
    ckpt_dir: Path,
    out_dir: Path,
    nbins: int,
    q_lo: float,
    q_hi: float,
    max_val_batches: Optional[int],
    plot_count: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta = _load_meta(cfg.SHARDS_DIR)

    # Prefer saved splits if present to guarantee you analyze the same held-out set you trained with.
    splits_path = ckpt_dir / "splits.npz"
    if splits_path.exists():
        sp = np.load(splits_path)
        train_idx = sp["train_idx"].astype(np.int64, copy=False)
        val_idx = sp["val_idx"].astype(np.int64, copy=False)
    else:
        train_idx, val_idx = _compute_splits_from_meta(meta, seed=cfg.SEED, val_fraction=cfg.VAL_FRACTION)

    _, dl_val = _make_loaders(
        cfg.SHARDS_DIR,
        train_idx=train_idx,
        val_idx=val_idx,
        num_workers=cfg.NUM_WORKERS,
        batch_size=cfg.BATCH_SIZE,
    )

    ckpts = _list_checkpoints(ckpt_dir)

    # Select a small subset of checkpoints for the r(z) plot (avoid unreadable spaghetti).
    if plot_count <= 0:
        plot_steps: set[int] = set()
    else:
        if len(ckpts) <= plot_count:
            plot_steps = set(range(len(ckpts)))
        else:
            idxs = np.linspace(0, len(ckpts) - 1, plot_count, dtype=int)
            plot_steps = set(int(i) for i in idxs)

    # Build model once, then just load weights each time (assumes consistent architecture).
    model = _build_model_from_cfg(device)

    metrics: List[LLRConsistency] = []
    curves: List[Tuple[int, np.ndarray, np.ndarray]] = []  # (step, x, r)

    for i, p in enumerate(ckpts):
        ck = torch.load(p, map_location="cpu")
        step = int(ck.get("step", -1))
        model.load_state_dict(ck["model"], strict=True)

        z, y, w = _infer_scores_on_val(model, dl_val, device, max_batches=max_val_batches)

        base, x, r = _llr_consistency_from_scores(
            z=z, y=y, w=w, nbins=int(nbins), q_lo=float(q_lo), q_hi=float(q_hi)
        )
        base.step = int(step)
        base.val_loss = (float(ck["val_loss"]) if ck.get("val_loss", None) is not None else None)
        metrics.append(base)

        if i in plot_steps:
            curves.append((int(step), x, r))

        print(
            f"[analyze] step={step:7d}  n_val={base.n_val:6d}  "
            f"auc_w={base.auc_w:.4f}  alpha={base.alpha:.3f}  beta={base.beta:.3f}  mse={base.mse:.3f}  "
            f"bins_used={base.n_bins_used:3d}  z_range=[{base.z_lo:.2f},{base.z_hi:.2f}]"
        )

    # Write metrics.csv
    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        wcsv = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "n_val",
                "auc_w",
                "alpha",
                "beta",
                "mse",
                "n_bins_used",
                "z_lo",
                "z_hi",
                "val_loss",
            ],
        )
        wcsv.writeheader()
        for m in metrics:
            wcsv.writerow(asdict(m))

    # Plot: AUC vs step
    steps = np.array([m.step for m in metrics], dtype=np.int64)
    aucs = np.array([m.auc_w for m in metrics], dtype=np.float64)
    plt.figure()
    plt.plot(steps, aucs, marker="o")
    plt.xlabel("training step")
    plt.ylabel("weighted AUC (class-conditional)")
    plt.title("AUC vs step")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "auc_vs_step.png", dpi=160)
    plt.close()

    # Plot: alpha, beta, mse vs step
    alphas = np.array([m.alpha for m in metrics], dtype=np.float64)
    betas = np.array([m.beta for m in metrics], dtype=np.float64)
    mses = np.array([m.mse for m in metrics], dtype=np.float64)

    plt.figure()
    plt.plot(steps, alphas, marker="o", label="alpha (slope)")
    plt.plot(steps, betas, marker="o", label="beta (intercept)")
    plt.xlabel("training step")
    plt.ylabel("fit params")
    plt.title("r(z) ≈ alpha*z + beta")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "alpha_beta_vs_step.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(steps, mses, marker="o")
    plt.xlabel("training step")
    plt.ylabel("MSE of (r(z)-z) over overlap bins")
    plt.title("LLR consistency error vs step")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "mse_vs_step.png", dpi=160)
    plt.close()

    # Plot: LLR consistency curves r(z) vs z (selected checkpoints)
    plt.figure()
    # Reference line y=x (on union range of plotted curves)
    if curves:
        allx = np.concatenate([c[1] for c in curves if c[1].size > 0], axis=0)
        if allx.size > 0:
            xmin = float(np.min(allx))
            xmax = float(np.max(allx))
            refx = np.linspace(xmin, xmax, 200)
            plt.plot(refx, refx, linestyle="--", label="y=x")

    curves.sort(key=lambda t: t[0])
    for step, x, r in curves:
        if x.size >= 2:
            plt.plot(x, r, marker=".", linewidth=1.0, label=f"step {step}")

    plt.xlabel("z (model logit)")
    plt.ylabel("r(z) = log p_s(z)/p_b(z) (weighted hist)")
    plt.title("LLR self-consistency in score space")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "llr_consistency.png", dpi=160)
    plt.close()

    print(f"[done] wrote {csv_path} and plots to {out_dir}/")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_train = sub.add_parser("train")
    ap_train.add_argument("--ckpt-dir", type=str, default="checkpoints")
    ap_train.add_argument("--max-steps", type=int, default=int(cfg.MAX_STEPS))
    ap_train.add_argument("--save-every", type=int, default=int(cfg.VAL_EVERY))
    ap_train.add_argument("--resume", action="store_true")

    ap_an = sub.add_parser("analyze")
    ap_an.add_argument("--ckpt-dir", type=str, default="checkpoints")
    ap_an.add_argument("--out-dir", type=str, default="llr_diagnostics")
    ap_an.add_argument("--nbins", type=int, default=60)
    ap_an.add_argument("--q-lo", type=float, default=0.005)
    ap_an.add_argument("--q-hi", type=float, default=0.995)
    ap_an.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="If set, cap how many val batches are used per checkpoint (for speed).",
    )
    ap_an.add_argument(
        "--plot-count",
        type=int,
        default=6,
        help="How many checkpoints to overlay in llr_consistency.png (evenly spaced).",
    )

    args = ap.parse_args()

    if args.cmd == "train":
        train_with_checkpoints(
            ckpt_dir=Path(args.ckpt_dir),
            max_steps=int(args.max_steps),
            save_every=int(args.save_every),
            resume=bool(args.resume),
        )
    elif args.cmd == "analyze":
        analyze_checkpoints(
            ckpt_dir=Path(args.ckpt_dir),
            out_dir=Path(args.out_dir),
            nbins=int(args.nbins),
            q_lo=float(args.q_lo),
            q_hi=float(args.q_hi),
            max_val_batches=(int(args.max_val_batches) if args.max_val_batches is not None else None),
            plot_count=int(args.plot_count),
        )
    else:
        raise RuntimeError(f"unknown cmd {args.cmd!r}")


if __name__ == "__main__":
    main()
