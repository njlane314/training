#!/usr/bin/env python3
"""
plot_attributions.py

Gradient-based attribution maps (per-plane) for the MultiViewSetClassifier.

What it does
------------
For a single event:
  - run a forward pass (with gradients enabled)
  - backprop a scalar objective (class-1: +logit, class-0: -logit)
  - compute per-sparse-site attribution via either:
        * gxi : |grad * input| reduced over channels
        * grad: |grad| reduced over channels
  - rasterize sparse coords -> dense 2D images for visualization
  - save a 2x3 grid:
        row 0: input intensity (sum over input channels)
        row 1: attribution magnitude (method above)
    columns correspond to planes: u, v, w

This file also enforces a 2-rows-by-3-columns layout (instead of 3x2).

Important note (why you see "missing attribution" on many hits)
---------------------------------------------------------------
The backbone uses MinkowskiGlobalMaxPooling. The gradient of a max is zero
for all non-argmax sites per channel, so gradient-based attributions are
inherently sparse: most input hits will have exactly zero attribution.
Use --explain-pool max_ste (recommended) or avg/lse to get dense maps.

Usage
-----
  python plot_attributions.py --ckpt checkpoints/ckpt_step0002000.pt --val-rank 0 --out-dir attrib
  python plot_attributions.py --ckpt checkpoints/ckpt_step0002000.pt --event-idx 12345 --target pred
  # random signal event (y=1) from the deterministic val split
  python plot_attributions.py --ckpt checkpoints/ckpt_step0002000.pt --random-signal --target true
  # random signal from anywhere (all good nnz>0 events if index.pt has nnz)
  python plot_attributions.py --ckpt checkpoints/ckpt_step0002000.pt --random-signal --random-from all --target true
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME

# Robust imports whether run as a script or as a module inside a package.
try:
    from . import config as cfg  # type: ignore
    from .dataset import ShardDataset, collate_me_fusion  # type: ignore
    from .fusion import MultiViewSetClassifier  # type: ignore
    from .model import make_backbone  # type: ignore
except Exception:
    import config as cfg  # type: ignore
    from dataset import ShardDataset, collate_me_fusion  # type: ignore
    from fusion import MultiViewSetClassifier  # type: ignore
    from model import make_backbone  # type: ignore


PLANES: Tuple[str, ...] = ("u", "v", "w")


def _load_meta(shards_dir: str) -> dict:
    return torch.load(f"{shards_dir}/index.pt", map_location="cpu")


def _compute_splits_from_meta(
    meta: dict,
    *,
    seed: int,
    val_fraction: float,
) -> Tuple[np.ndarray, np.ndarray]:
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
            raise ValueError("All events have nnz==0 after sparsification.")
        perm = rng.permutation(idx_all.size)
        idx_perm = idx_all[perm]
        n_val = int(float(val_fraction) * idx_perm.size)
        val_idx = idx_perm[:n_val]
        train_idx = idx_perm[n_val:]
        return train_idx.astype(np.int64), val_idx.astype(np.int64)

    perm = rng.permutation(n)
    n_val = int(float(val_fraction) * n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


def _sort_event_indices_for_io(meta: dict, event_idx: np.ndarray) -> np.ndarray:
    shard_events = int(meta.get("shard_events", getattr(cfg, "SHARD_EVENTS", 2048)))
    shard_id = (event_idx // shard_events).astype(np.int64, copy=False)
    local_id = (event_idx - shard_id * shard_events).astype(np.int64, copy=False)
    key = np.lexsort((local_id, shard_id))
    return event_idx[key].astype(np.int64, copy=False)


def _build_model_from_cfg(device: torch.device) -> nn.Module:
    backbone = make_backbone(cfg.BACKBONE, in_ch=2, embed_dim=cfg.EMBED_DIM).to(device)
    model = MultiViewSetClassifier(backbone=backbone, embed_dim=cfg.EMBED_DIM, plane_names=PLANES).to(device)
    return model


def _step_to_inputs_with_grads(
    coords_by_plane: Dict[str, torch.Tensor],
    feats_by_plane: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[Dict[str, ME.SparseTensor], Dict[str, torch.Tensor]]:
    """
    Returns:
      inputs: Dict[str, SparseTensor]
      feats_leaf: Dict[str, torch.Tensor]  (leaf tensors with requires_grad=True)
    """
    inputs: Dict[str, ME.SparseTensor] = {}
    feats_leaf: Dict[str, torch.Tensor] = {}
    for name in PLANES:
        # Make a true leaf tensor so gradients are always accessible.
        feats = feats_by_plane[name].to(device, non_blocking=True).detach().requires_grad_(True)
        coords = coords_by_plane[name]  # keep CPU int32
        inputs[name] = ME.SparseTensor(features=feats, coordinates=coords, device=device)
        feats_leaf[name] = feats
    return inputs, feats_leaf


def _sparse_to_dense_2d(
    coords_byx: np.ndarray,
    values: np.ndarray,
    *,
    out_shape: Optional[Tuple[int, int]] = None,
    batch_index: int = 0,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    coords_byx: [N,3] (batch,y,x)  (matches process.py + collate_me_fusion)
    values:     [N]

    Returns:
      img:    [H,W] with y as rows, x as cols (imshow origin='lower' friendly)
      extent: (xmin, xmax, ymin, ymax) for matplotlib imshow
    """
    if coords_byx.size == 0:
        img = np.zeros((1, 1), dtype=np.float32)
        return img, (0.0, 1.0, 0.0, 1.0)

    c = np.asarray(coords_byx)
    if c.ndim != 2 or c.shape[1] != 3:
        raise ValueError(f"coords must have shape [N,3], got {c.shape}")

    b = c[:, 0].astype(np.int64, copy=False)
    sel = (b == int(batch_index))
    if not np.any(sel):
        img = np.zeros((1, 1), dtype=np.float32)
        return img, (0.0, 1.0, 0.0, 1.0)

    # IMPORTANT: (batch, y, x)
    y = c[sel, 1].astype(np.int64, copy=False)
    x = c[sel, 2].astype(np.int64, copy=False)
    v = np.asarray(values, dtype=np.float32).reshape(-1)[sel]

    if out_shape is not None:
        H, W = int(out_shape[0]), int(out_shape[1])
        img = np.zeros((H, W), dtype=np.float32)
        inb = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        if np.any(inb):
            np.add.at(img, (y[inb], x[inb]), v[inb])
        extent = (-0.5, float(W) - 0.5, -0.5, float(H) - 0.5)
        return img, extent

    xmin, xmax = int(x.min()), int(x.max())
    ymin, ymax = int(y.min()), int(y.max())
    w = int(xmax - xmin + 1)
    h = int(ymax - ymin + 1)
    img = np.zeros((h, w), dtype=np.float32)
    np.add.at(img, (y - ymin, x - xmin), v)
    extent = (float(xmin) - 0.5, float(xmax) + 0.5, float(ymin) - 0.5, float(ymax) + 0.5)
    return img, extent


def _nonzero_percentiles(
    img: np.ndarray,
    *,
    q_lo: float,
    q_hi: float,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Percentiles over strictly-positive pixels (zeros are "no data" for LogNorm).
    Returns (vlo, vhi) or (None, None) if no positive pixels exist.
    """
    nz = np.asarray(img, dtype=np.float32).reshape(-1)
    nz = nz[nz > 0]
    if nz.size == 0:
        return None, None
    lo = float(np.percentile(nz, float(q_lo)))
    hi = float(np.percentile(nz, float(q_hi)))
    return lo, hi


class _SparseGlobalLogSumExpPool(nn.Module):
    """
    Global log-sum-exp pooling over sparse sites, per batch and per channel.
    Returns an ME.SparseTensor with one coordinate per batch at (b,0,0).
    """

    def __init__(self, beta: float):
        super().__init__()
        beta = float(beta)
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError(f"beta must be > 0, got {beta}")
        self.beta = beta

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        F_in = x.F
        if F_in.numel() == 0:
            # Degenerate; return empty batch.
            coords_out = torch.zeros((0, x.C.shape[1]), dtype=torch.int32)
            return ME.SparseTensor(features=F_in, coordinates=coords_out, device=F_in.device)

        # batch indices: (N,) on GPU for masking
        b = x.C[:, 0].to(device=F_in.device, dtype=torch.int64, non_blocking=True)
        B = int(b.max().item()) + 1
        C = int(F_in.shape[1])

        out = torch.zeros((B, C), device=F_in.device, dtype=F_in.dtype)
        beta = float(self.beta)

        for bi in range(B):
            m = (b == bi)
            if m.any():
                Fb = F_in[m]  # [Ni, C]
                out[bi] = torch.logsumexp(beta * Fb, dim=0) / beta

        coords_out = torch.zeros((B, x.C.shape[1]), dtype=torch.int32)  # CPU coords
        coords_out[:, 0] = torch.arange(B, dtype=torch.int32)
        return ME.SparseTensor(features=out, coordinates=coords_out, device=out.device)


class _SparseGlobalMaxPoolSTE(nn.Module):
    """
    Max pooling forward, but use log-sum-exp gradient (straight-through estimator).

    Forward value matches true max pooling:
        y = max(F)
    Backward uses smooth LSE gradient:
        dy/dF ~ softmax(beta * F)

    This is ideal for attribution:
      - logits match the checkpoint (same forward as max pooling)
      - gradients are dense instead of argmax-sparse
    """

    def __init__(self, beta: float):
        super().__init__()
        beta = float(beta)
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError(f"beta must be > 0, got {beta}")
        self.beta = beta

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        F_in = x.F
        if F_in.numel() == 0:
            coords_out = torch.zeros((0, x.C.shape[1]), dtype=torch.int32)
            return ME.SparseTensor(features=F_in, coordinates=coords_out, device=F_in.device)

        b = x.C[:, 0].to(device=F_in.device, dtype=torch.int64, non_blocking=True)
        B = int(b.max().item()) + 1
        C = int(F_in.shape[1])

        out = torch.zeros((B, C), device=F_in.device, dtype=F_in.dtype)
        beta = float(self.beta)

        for bi in range(B):
            m = (b == bi)
            if m.any():
                Fb = F_in[m]  # [Ni, C]
                maxv = Fb.max(dim=0).values
                lse = torch.logsumexp(beta * Fb, dim=0) / beta
                # Forward == maxv, gradient == grad(lse)
                out[bi] = lse + (maxv - lse).detach()

        coords_out = torch.zeros((B, x.C.shape[1]), dtype=torch.int32)
        coords_out[:, 0] = torch.arange(B, dtype=torch.int32)
        return ME.SparseTensor(features=out, coordinates=coords_out, device=out.device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Checkpoint path (ckpt_step*.pt).")
    ap.add_argument("--out-dir", type=str, default="attrib")
    ap.add_argument("--out-name", type=str, default="attributions.png")

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--event-idx", type=int, default=None, help="Global event index to visualize.")
    g.add_argument(
        "--val-rank",
        type=int,
        default=0,
        help="If --event-idx not set: pick this rank from the deterministic val split.",
    )
    g.add_argument(
        "--random",
        action="store_true",
        help="If --event-idx not set: pick a random event from --random-from (default: val).",
    )
    g.add_argument(
        "--random-signal",
        action="store_true",
        help="If --event-idx not set: pick a random signal event (y=1) from --random-from (default: val).",
    )
    g.add_argument(
        "--random-background",
        action="store_true",
        help="If --event-idx not set: pick a random background event (y=0) from --random-from (default: val).",
    )

    ap.add_argument(
        "--random-from",
        type=str,
        default="val",
        choices=("val", "train", "all"),
        help="Pool to draw random events from (used with --random*).",
    )
    ap.add_argument(
        "--rng-seed",
        type=int,
        default=None,
        help="Seed for random event selection (omit for nondeterministic selection).",
    )

    ap.add_argument(
        "--target",
        type=str,
        default="pred",
        choices=("pred", "true", "1", "0"),
        help="Which class to attribute: predicted, true label, or fixed {1,0}. "
        "For class-0 attributions we backprop -logit.",
    )

    ap.add_argument(
        "--attrib",
        type=str,
        default="gxi",
        choices=("gxi", "grad"),
        help="Attribution method per sparse site: gxi=|grad*input|, grad=|grad| (both reduced over channels).",
    )
    ap.add_argument(
        "--attr-norm",
        type=str,
        default="log",
        choices=("log", "linear"),
        help=(
            "How to map attribution magnitudes to colors. "
            "'log' uses matplotlib LogNorm on nonzero pixels with percentile vmin/vmax (recommended). "
            "'linear' uses linear scaling with vmax from a high percentile."
        ),
    )
    ap.add_argument(
        "--explain-pool",
        type=str,
        default="max_ste",
        choices=("max", "max_ste", "avg", "lse"),
        help=(
            "Global pooling used inside the backbone during the attribution run. "
            "max gives true gradients but is argmax-sparse (many hits get 0 attribution). "
            "max_ste keeps the *forward* identical to max pooling but uses a smooth LSE gradient (recommended). "
            "avg/lse use smooth pooling in both forward+backward."
        ),
    )
    ap.add_argument(
        "--pool-beta",
        type=float,
        default=5.0,
        help="Beta for LSE pooling / max_ste surrogate gradient. Smaller => denser gradients; larger => closer to hard max.",
    )
    ap.add_argument(
        "--attr-qlo",
        type=float,
        default=5.0,
        help="Lower percentile for attribution color scaling (computed on nonzero pixels). Used for LogNorm vmin.",
    )
    ap.add_argument(
        "--attr-qhi",
        type=float,
        default=99.5,
        help="Upper percentile for attribution color scaling (computed on nonzero pixels). Used for vmax.",
    )

    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=0)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / str(args.out_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load metadata once (also gives authoritative H/W used when shards were made).
    meta = _load_meta(cfg.SHARDS_DIR)
    H = int(meta.get("H", getattr(cfg, "H", 512)))
    W = int(meta.get("W", getattr(cfg, "W", 512)))

    # Pick an event.
    if args.event_idx is not None:
        event_idx = int(args.event_idx)
    else:
        # Build deterministic splits (matches training).
        train_idx, val_idx = _compute_splits_from_meta(meta, seed=int(cfg.SEED), val_fraction=float(cfg.VAL_FRACTION))

        # Build "all good events" pool (respect nnz>0 filtering if present).
        n_events = int(meta["n_events"])
        labels_all = np.asarray(meta["labels"], dtype=np.uint8).reshape(-1)
        if labels_all.shape[0] != n_events:
            raise ValueError(f"index.pt labels len={labels_all.shape[0]} != n_events={n_events}")

        good_idx: np.ndarray
        if "nnz" in meta and meta["nnz"] is not None:
            if isinstance(meta["nnz"], torch.Tensor):
                nnz_all = meta["nnz"].to(dtype=torch.int64).cpu().numpy().reshape(-1)
            else:
                nnz_all = np.asarray(meta["nnz"], dtype=np.int64).reshape(-1)
            if nnz_all.shape[0] != n_events:
                raise ValueError(f"index.pt nnz len={nnz_all.shape[0]} != n_events={n_events}")
            good_idx = np.flatnonzero(nnz_all > 0).astype(np.int64, copy=False)
        else:
            good_idx = np.arange(n_events, dtype=np.int64)

        do_random = bool(args.random or args.random_signal or args.random_background)
        if do_random:
            if args.random_from == "val":
                pool = val_idx
            elif args.random_from == "train":
                pool = train_idx
            else:
                pool = good_idx

            want_label: Optional[int] = None
            if args.random_signal:
                want_label = 1
            elif args.random_background:
                want_label = 0

            if want_label is not None:
                pool = pool[labels_all[pool] == np.uint8(want_label)]

            if pool.size == 0:
                raise ValueError(
                    f"No events available for random selection: random_from={args.random_from} "
                    f"label={want_label if want_label is not None else 'any'}"
                )

            rng = np.random.default_rng(None if args.rng_seed is None else int(args.rng_seed))
            event_idx = int(rng.choice(pool))
            y_meta = int(labels_all[event_idx])
            sel = "random"
            if args.random_signal:
                sel = "random-signal"
            elif args.random_background:
                sel = "random-background"
            print(f"[pick] {sel} from {args.random_from}: event_idx={event_idx} (y={y_meta}) seed={args.rng_seed}")
        else:
            # Deterministic val-rank selection (IO-sorted for convenience).
            val_idx = _sort_event_indices_for_io(meta, val_idx)
            vr = int(args.val_rank)
            if vr < 0 or vr >= int(val_idx.size):
                raise ValueError(f"--val-rank out of range: {vr} (val size={val_idx.size})")
            event_idx = int(val_idx[vr])

    # Load model + checkpoint.
    model = _build_model_from_cfg(device)
    ck = torch.load(args.ckpt, map_location="cpu")
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(state, strict=True)
    model.eval()

    # Override backbone global pooling for attribution behavior.
    # This does not affect checkpoint loading (pool has no parameters).
    if hasattr(model, "backbone") and hasattr(model.backbone, "pool"):
        if args.explain_pool == "avg":
            model.backbone.pool = ME.MinkowskiGlobalAvgPooling()
        elif args.explain_pool == "lse":
            model.backbone.pool = _SparseGlobalLogSumExpPool(beta=float(args.pool_beta))
        elif args.explain_pool == "max_ste":
            model.backbone.pool = _SparseGlobalMaxPoolSTE(beta=float(args.pool_beta))
        else:
            # "max": leave as-is
            pass
    else:
        raise RuntimeError("Expected model.backbone.pool to exist; cannot override pooling for attribution.")

    # Load the single event through the same collate path as training/inference.
    ds = ShardDataset(cfg.SHARDS_DIR, np.asarray([event_idx], dtype=np.int64), cache_size=2)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate_me_fusion,
        pin_memory=True,
        persistent_workers=False,
    )

    try:
        batch = next(iter(dl))
    except StopIteration as exc:
        raise RuntimeError("Empty DataLoader for the requested event?") from exc

    coords_by_plane, feats_by_plane, y, available_mask = batch
    y0 = int(y[0].item()) if hasattr(y, "numel") and y.numel() > 0 else -1

    # Build inputs with leaf tensors requiring gradients.
    inputs, feats_leaf = _step_to_inputs_with_grads(coords_by_plane, feats_by_plane, device)
    m = available_mask.to(device, non_blocking=True)

    # Forward + choose scalar objective.
    # Disable AMP for attribution stability.
    with torch.cuda.amp.autocast(enabled=False):
        logits = model(inputs, available_mask=m).squeeze(1)  # [B]
        if logits.numel() != 1:
            raise RuntimeError(f"Expected batch_size=1 for attribution, got logits shape {tuple(logits.shape)}")
        logit = logits[0]

        if args.target == "pred":
            target_class = int((logit.detach() > 0).item())
        elif args.target == "true":
            target_class = int(y0)
        elif args.target == "1":
            target_class = 1
        else:
            target_class = 0

        objective = logit if target_class == 1 else -logit

    # Compute grads explicitly (more reliable than relying on .grad side effects).
    feat_list = [feats_leaf[name] for name in PLANES]
    grad_list = torch.autograd.grad(
        outputs=objective,
        inputs=feat_list,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    grads_by_plane: Dict[str, Optional[torch.Tensor]] = dict(zip(PLANES, grad_list))

    # Plot: 2 rows x 3 cols (u,v,w). Top=input, bottom=attribution.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fig, axs = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)

    for ci, name in enumerate(PLANES):
        ax_in = axs[0, ci]
        ax_at = axs[1, ci]

        is_avail = bool(available_mask[0, ci].item())
        if not is_avail:
            ax_in.axis("off")
            ax_at.axis("off")
            ax_in.text(0.5, 0.5, f"{name}: missing", ha="center", va="center", transform=ax_in.transAxes)
            continue

        coords = coords_by_plane[name].detach().cpu().numpy()
        feats = feats_leaf[name].detach()
        grad = grads_by_plane.get(name, None)
        if grad is None:
            # Disconnected (should be rare). Treat as zero attribution.
            grad = torch.zeros_like(feats)
        else:
            grad = grad.detach()

        # How sparse are gradients at the input sites?
        g_site = grad.abs().sum(dim=1)
        frac_nz = float((g_site > 0).to(dtype=torch.float32).mean().cpu())

        # Per-site scalars.
        inp_val = feats.sum(dim=1).detach().cpu().numpy()
        if args.attrib == "grad":
            attr_val = grad.abs().sum(dim=1).detach().cpu().numpy()
        else:
            attr_val = (grad * feats).abs().sum(dim=1).detach().cpu().numpy()

        # Debug: if this prints zeros, the model is locally insensitive to the inputs for this event.
        gmax = float(grad.abs().max().cpu())
        amax = float(np.max(attr_val)) if attr_val.size else 0.0
        nz = attr_val[attr_val > 0]
        if nz.size:
            p50, p90, p99 = np.percentile(nz, [50.0, 90.0, 99.0])
            amin_nz = float(nz.min())
        else:
            amin_nz, p50, p90, p99 = 0.0, 0.0, 0.0, 0.0
        print(
            f"[attrib] plane={name}  n_sites={int(feats.shape[0])}  "
            f"grad_nz_frac={frac_nz:.3f}  explain_pool={args.explain_pool}  "
            f"grad_abs_max={gmax:.3e}  attr_nz_min={amin_nz:.3e}  "
            f"attr_p50={p50:.3e} attr_p90={p90:.3e} attr_p99={p99:.3e} attr_max={amax:.3e}"
        )

        # coords are (batch, y, x); rasterize into the full detector frame by default.
        img_in, extent_in = _sparse_to_dense_2d(coords, inp_val, out_shape=(H, W), batch_index=0)
        img_at, extent_at = _sparse_to_dense_2d(coords, attr_val, out_shape=(H, W), batch_index=0)

        ax_in.imshow(img_in, origin="lower", extent=extent_in, aspect="auto", interpolation="nearest")
        ax_in.set_title(f"{name} input")
        ax_in.set_xlabel("x")
        ax_in.set_ylabel("y")

        # Attribution visualization:
        # - For values in [1e-5, 1e-1], log1p is ~identity and hides structure.
        # - Use LogNorm over nonzero pixels, matching your earlier decade-scaled plots.
        img_at_f = img_at.astype(np.float32, copy=False)
        if args.attr_norm == "log":
            vlo, vhi = _nonzero_percentiles(img_at_f, q_lo=float(args.attr_qlo), q_hi=float(args.attr_qhi))
            if vlo is None or vhi is None:
                ax_at.imshow(img_at_f, origin="lower", extent=extent_at, aspect="auto", interpolation="nearest")
            else:
                # Guard against vmin<=0 and vhi<=vlo.
                vmin = max(float(vlo), 1e-12)
                vmax = max(float(vhi), vmin * 1.001)
                show = np.ma.masked_less_equal(img_at_f, 0.0)
                ax_at.imshow(
                    show,
                    origin="lower",
                    extent=extent_at,
                    aspect="auto",
                    interpolation="nearest",
                    norm=LogNorm(vmin=vmin, vmax=vmax),
                )
        else:
            # Linear scaling but ignore zeros when picking vmax.
            _, vhi = _nonzero_percentiles(img_at_f, q_lo=50.0, q_hi=float(args.attr_qhi))
            ax_at.imshow(
                img_at_f,
                origin="lower",
                extent=extent_at,
                aspect="auto",
                interpolation="nearest",
                vmin=0.0,
                vmax=vhi,
            )

        ax_at.set_title(f"{name} attribution ({args.attrib})  norm={args.attr_norm}")
        ax_at.set_xlabel("x")
        ax_at.set_ylabel("y")

    fig.suptitle(
        f"event_idx={event_idx}  y={y0}  target={target_class}  "
        f"logit={float(logit.detach().cpu()):+.3f}  explain_pool={args.explain_pool} beta={float(args.pool_beta):g}"
    )
    fig.savefig(str(out_path), dpi=160)
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
