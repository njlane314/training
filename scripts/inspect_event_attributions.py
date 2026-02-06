#!/usr/bin/env python3
"""
inspect_event_attributions.py

Pick a random *high-score* event (likely signal) and plot:
  - per-plane event display (log-charge feature)
  - per-plane attribution overlay (default: grad*input on the model logit)

Usage example:
  python scripts/inspect_event_attributions.py \
      --shards_dir shards \
      --checkpoint checkpoints/checkpoint.pt \
      --num_samples 5000 --min_score 0.95 \
      --crop --out high_score_event.png

Notes
-----
- Event display uses the model input representation: sparse hits expanded to an HxW image,
  with pixel intensity = feature[channel] (default channel=logq = log1p(charge)).
- Attribution uses gradients w.r.t. the input features at each sparse site and maps
  them back to pixel coordinates.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import torch
import torch.utils.data

import MinkowskiEngine as ME

from likelihood import config as cfg
from likelihood.dataset import ShardDataset, collate_me_fusion
from likelihood.fusion import MultiViewSetClassifier
from likelihood.model import make_backbone


PLANES: Tuple[str, ...] = ("u", "v", "w")
_STEP_RE = re.compile(r"_step(\d+)\.pt$")


# -------------------------
# Checkpoint helpers
# -------------------------
def resolve_checkpoint(path: str) -> Path:
    """
    Resolve a checkpoint argument.

    Accepts:
      - an existing .pt file
      - a base path like checkpoints/checkpoint.pt (will pick newest *_step*.pt)
      - a directory (will pick newest *_step*.pt or newest .pt by mtime)
    """
    p = Path(path)

    if p.is_file():
        return p

    if p.is_dir():
        step_files = []
        for q in p.glob("*.pt"):
            m = _STEP_RE.search(q.name)
            if m:
                step_files.append((int(m.group(1)), q))
        if step_files:
            return max(step_files, key=lambda t: t[0])[1]

        pts = list(p.glob("*.pt"))
        if pts:
            return max(pts, key=lambda q: q.stat().st_mtime)

        raise FileNotFoundError(f"No .pt checkpoints found in directory: {p}")

    if p.suffix == ".pt":
        d = p.parent if str(p.parent) != "" else Path(".")
        stem = p.stem
        step_files = []
        for q in d.glob(f"{stem}_step*.pt"):
            m = _STEP_RE.search(q.name)
            if m:
                step_files.append((int(m.group(1)), q))
        if step_files:
            return max(step_files, key=lambda t: t[0])[1]

    raise FileNotFoundError(
        f"Checkpoint not found: {p} (also tried resolving *_step*.pt variants)."
    )


# -------------------------
# Dataset helpers
# -------------------------
class ShardDatasetWithID(ShardDataset):
    """ShardDataset that also returns the global event id (gi)."""

    def __getitem__(self, i: int):
        coords, feats, y = super().__getitem__(i)
        gi = int(self.event_indices[i])
        return coords, feats, y, gi


def collate_me_fusion_with_id(batch, plane_names: Tuple[str, ...] = PLANES):
    """collate_me_fusion + return global ids tensor."""
    coords_feats_y = [(c, f, y) for (c, f, y, _gi) in batch]
    coords, feats, y, available_mask = collate_me_fusion(coords_feats_y, plane_names=plane_names)
    gids = torch.tensor([gi for (_c, _f, _y, gi) in batch], dtype=torch.int64)
    return coords, feats, y, available_mask, gids


class FixedOrderSampler(torch.utils.data.Sampler[int]):
    """Yield exactly the provided indices, in that order."""

    def __init__(self, indices: Sequence[int]):
        self.indices = list(map(int, indices))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


# -------------------------
# Scoring / selection
# -------------------------
@torch.no_grad()
def score_event_batch(
    model: torch.nn.Module,
    coords_by_plane: Dict[str, torch.Tensor],
    feats_by_plane: Dict[str, torch.Tensor],
    available_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Forward a batch and return sigmoid scores, shape [B]."""
    inputs: Dict[str, ME.SparseTensor] = {}
    for name in PLANES:
        feats = feats_by_plane[name].to(device, non_blocking=True)
        coords = coords_by_plane[name]  # CPU int32
        inputs[name] = ME.SparseTensor(features=feats, coordinates=coords, device=device)

    logits = model(inputs, available_mask=available_mask.to(device, non_blocking=True)).squeeze(1)
    return torch.sigmoid(logits)


def pick_random_high_score_event(
    model: torch.nn.Module,
    shards_dir: str,
    *,
    num_samples: int,
    min_score: float,
    topk_fallback: int,
    seed: Optional[int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    restrict_to_nnz_gt0: bool = True,
    require_label: Optional[int] = None,
) -> Tuple[int, float, Dict]:
    """
    Sample `num_samples` events, score them, and pick a random event among those with score>=min_score.
    If none exceed min_score, pick random among the top `topk_fallback` within the sample.

    Returns: (global_event_id, score, meta_dict)
    """
    meta = torch.load(f"{shards_dir}/index.pt", map_location="cpu")
    n_events = int(meta["n_events"])
    labels_all = np.asarray(meta["labels"], dtype=np.uint8).reshape(-1)
    if labels_all.shape[0] != n_events:
        raise ValueError(f"index.pt labels len={labels_all.shape[0]} != n_events={n_events}")

    h_meta = int(meta.get("H", getattr(cfg, "H", 512)))
    w_meta = int(meta.get("W", getattr(cfg, "W", 512)))

    pool = np.arange(n_events, dtype=np.int64)
    if restrict_to_nnz_gt0 and ("nnz" in meta) and (meta["nnz"] is not None):
        nnz = meta["nnz"]
        if isinstance(nnz, torch.Tensor):
            nnz = nnz.to(dtype=torch.int64).cpu().numpy()
        else:
            nnz = np.asarray(nnz, dtype=np.int64)
        nnz = nnz.reshape(-1)
        if nnz.shape[0] == n_events:
            pool = np.flatnonzero(nnz > 0).astype(np.int64, copy=False)

    if require_label is not None:
        label_mask = labels_all == int(require_label)
        if label_mask.shape[0] == n_events:
            pool = pool[label_mask[pool]]

    if pool.size == 0:
        raise ValueError("No eligible events in the pool to sample from (pool.size==0).")

    rng = np.random.default_rng(None if seed is None else int(seed))
    k = int(min(num_samples, pool.size))
    chosen = rng.choice(pool, size=k, replace=False)  # global ids

    ds = ShardDatasetWithID(shards_dir, chosen, cache_size=2)

    # IO-friendly order: sort by (shard_id, local_id)
    shard_ids = (chosen // ds.shard_events).astype(np.int64, copy=False)
    local_ids = (chosen - shard_ids * ds.shard_events).astype(np.int64, copy=False)
    order = np.lexsort((local_ids, shard_ids))  # dataset indices in desired order

    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=int(batch_size),
        sampler=FixedOrderSampler(order.tolist()),
        num_workers=int(num_workers),
        collate_fn=collate_me_fusion_with_id,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(int(num_workers) > 0),
    )

    scored: List[Tuple[int, float]] = []
    for coords_by_plane, feats_by_plane, _y, available_mask, gids in dl:
        p = score_event_batch(model, coords_by_plane, feats_by_plane, available_mask, device=device)
        for gi, pi in zip(gids.tolist(), p.detach().cpu().tolist()):
            scored.append((int(gi), float(pi)))

    if not scored:
        raise RuntimeError("Did not score any events (empty scored list).")

    high = [(gi, s) for (gi, s) in scored if s >= float(min_score)]
    if high:
        gi, s = high[int(rng.integers(len(high)))]
        sel = "threshold"
    else:
        scored_sorted = sorted(scored, key=lambda t: t[1], reverse=True)
        kk = int(min(topk_fallback, len(scored_sorted)))
        if kk <= 0:
            kk = 1
        subset = scored_sorted[:kk]
        gi, s = subset[int(rng.integers(len(subset)))]
        sel = f"top{kk}"

    meta_out = {
        "n_scored": len(scored),
        "n_high": len(high),
        "selection": sel,
        "true_label": int(labels_all[int(gi)]) if 0 <= int(gi) < labels_all.shape[0] else None,
        "H": h_meta,
        "W": w_meta,
    }
    return int(gi), float(s), meta_out


# -------------------------
# Attribution
# -------------------------
def _dense_from_sparse(
    coords: torch.Tensor,
    values: torch.Tensor,
    *,
    h: int,
    w: int,
) -> np.ndarray:
    """
    coords: int tensor [N,3] with (batch,y,x). Uses only y,x (assumes batch==0).
    values: float tensor [N]
    """
    yy = coords[:, 1].to(torch.int64).cpu().numpy()
    xx = coords[:, 2].to(torch.int64).cpu().numpy()
    vv = values.detach().cpu().numpy().astype(np.float32, copy=False)

    img = np.zeros((int(h), int(w)), dtype=np.float32)
    np.add.at(img, (yy, xx), vv)
    return img


def compute_gradxinput_attribution(
    model: torch.nn.Module,
    coords: np.ndarray,
    feats: np.ndarray,
    *,
    device: torch.device,
    h: int,
    w: int,
    channel: int = 1,
    signed: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], float]:
    """
    Compute per-plane attribution = grad(logit)/d(feature[channel]) * feature[channel].

    Returns:
      - images: dict plane -> dense HxW image (feature[channel])
      - attrs : dict plane -> dense HxW attribution map
      - score : sigmoid(logit) float
    """
    coords_by_plane, feats_by_plane, _y, available_mask = collate_me_fusion(
        [(coords, feats, 0.0)],
        plane_names=PLANES,
    )

    feat_leaf: Dict[str, torch.Tensor] = {}
    inputs: Dict[str, ME.SparseTensor] = {}
    for name in PLANES:
        F = feats_by_plane[name].to(device).detach().requires_grad_(True)  # leaf
        C = coords_by_plane[name]  # CPU int32
        feat_leaf[name] = F
        inputs[name] = ME.SparseTensor(features=F, coordinates=C, device=device)

    model.zero_grad(set_to_none=True)

    logit = model(inputs, available_mask=available_mask.to(device)).view(-1)[0]
    score = torch.sigmoid(logit).detach().cpu().item()

    logit.backward()

    images: Dict[str, np.ndarray] = {}
    attrs: Dict[str, np.ndarray] = {}

    for name in PLANES:
        C = coords_by_plane[name]
        F = feat_leaf[name].detach()
        G = feat_leaf[name].grad
        if G is None:
            raise RuntimeError(f"No gradient for plane {name}; check requires_grad setup.")

        f = F[:, int(channel)]
        g = G[:, int(channel)]
        a = g * f
        if not signed:
            a = torch.clamp(a, min=0.0)

        images[name] = _dense_from_sparse(C, f, h=h, w=w)
        attrs[name] = _dense_from_sparse(C, a, h=h, w=w)

    return images, attrs, float(score)


def compute_event_images(
    coords: np.ndarray,
    feats: np.ndarray,
    *,
    h: int,
    w: int,
    channel: int = 1,
) -> Dict[str, np.ndarray]:
    """
    Compute per-plane dense images for a given feature channel.

    Returns:
      - images: dict plane -> dense HxW image (feature[channel])
    """
    coords_by_plane, feats_by_plane, _y, _available_mask = collate_me_fusion(
        [(coords, feats, 0.0)],
        plane_names=PLANES,
    )

    images: Dict[str, np.ndarray] = {}
    for name in PLANES:
        C = coords_by_plane[name]
        F = feats_by_plane[name]
        f = F[:, int(channel)]
        images[name] = _dense_from_sparse(C, f, h=h, w=w)

    return images


# -------------------------
# Plotting
# -------------------------
def _auto_vmax(x: np.ndarray, q: float = 99.5) -> Optional[float]:
    flat = x[np.isfinite(x)]
    flat = flat[flat != 0]
    if flat.size == 0:
        return None
    return float(np.percentile(flat, q))


def _attr_log_norm(att: np.ndarray) -> Optional[colors.Normalize]:
    flat = att[np.isfinite(att)]
    if flat.size == 0:
        return None
    if np.any(flat < 0):
        vmax = float(np.max(np.abs(flat)))
        if vmax <= 0:
            return None
        linthresh = max(vmax * 1e-3, 1e-6)
        return colors.SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax)

    positive = flat[flat > 0]
    if positive.size == 0:
        return None
    vmin = float(np.percentile(positive, 1.0))
    vmax = float(np.percentile(positive, 99.5))
    if vmin <= 0:
        vmin = float(np.min(positive))
    if vmax <= vmin:
        vmax = float(np.max(positive))
    return colors.LogNorm(vmin=vmin, vmax=vmax)


def _global_log_norm(arrs: Sequence[np.ndarray], *, signed: bool) -> Optional[colors.Normalize]:
    flat = np.concatenate([a[np.isfinite(a)].reshape(-1) for a in arrs])
    if flat.size == 0:
        return None
    if signed and np.any(flat < 0):
        vmax = float(np.max(np.abs(flat)))
        if vmax <= 0:
            return None
        linthresh = max(vmax * 1e-3, 1e-6)
        return colors.SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax)

    positive = flat[flat > 0]
    if positive.size == 0:
        return None
    vmin = float(np.percentile(positive, 1.0))
    vmax = float(np.percentile(positive, 99.5))
    if vmin <= 0:
        vmin = float(np.min(positive))
    if vmax <= vmin:
        vmax = float(np.max(positive))
    return colors.LogNorm(vmin=vmin, vmax=vmax)


def _crop_box_from_planes(imgs: Dict[str, np.ndarray], margin: int) -> Optional[Tuple[slice, slice]]:
    ys: List[int] = []
    xs: List[int] = []
    for arr in imgs.values():
        yy, xx = np.nonzero(arr)
        if yy.size:
            ys.extend(yy.tolist())
            xs.extend(xx.tolist())
    if not ys:
        return None
    h, w = next(iter(imgs.values())).shape
    y0 = max(min(ys) - int(margin), 0)
    y1 = min(max(ys) + int(margin) + 1, h)
    x0 = max(min(xs) - int(margin), 0)
    x1 = min(max(xs) + int(margin) + 1, w)
    return (slice(y0, y1), slice(x0, x1))


def plot_event_and_attribution(
    images: Dict[str, np.ndarray],
    attrs: Dict[str, np.ndarray],
    *,
    out: Optional[str],
    title: str,
    crop: bool,
    crop_margin: int,
) -> None:
    crop_slices: Optional[Tuple[slice, slice]] = None
    if crop:
        crop_slices = _crop_box_from_planes(images, margin=crop_margin)

    img_views: List[np.ndarray] = []
    attr_views: List[np.ndarray] = []
    for name in PLANES:
        img = images[name]
        att = attrs[name]
        if crop_slices is not None:
            img = img[crop_slices]
            att = att[crop_slices]
        img_views.append(img)
        attr_views.append(att)

    img_norm = _global_log_norm(img_views, signed=False)
    attr_norm = _global_log_norm(attr_views, signed=True)

    fig, axes = plt.subplots(nrows=len(PLANES), ncols=2, figsize=(12, 15), constrained_layout=True)
    fig.suptitle(title)

    for i, name in enumerate(PLANES):
        img = img_views[i]
        att = attr_views[i]

        vmax0 = _auto_vmax(img)
        vmax1 = _auto_vmax(att)

        ax0 = axes[i, 0]
        ax1 = axes[i, 1]

        im0 = ax0.imshow(
            img,
            origin="lower",
            vmax=None if img_norm is not None else vmax0,
            interpolation="nearest",
            norm=img_norm,
        )
        ax0.set_title(f"{name}: event (feature)")
        ax0.set_xlabel("x")
        ax0.set_ylabel("y")
        fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

        ax1.imshow(
            img,
            origin="lower",
            cmap="gray",
            vmax=None if img_norm is not None else vmax0,
            interpolation="nearest",
            norm=img_norm,
        )
        im1 = ax1.imshow(
            att,
            origin="lower",
            alpha=0.6,
            vmax=None if attr_norm is not None else vmax1,
            interpolation="nearest",
            norm=attr_norm,
        )
        ax1.set_title(f"{name}: attribution overlay (grad×input)")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"[saved] {out_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_event_only(
    images: Dict[str, np.ndarray],
    *,
    out: Optional[str],
    title: str,
    crop: bool,
    crop_margin: int,
) -> None:
    crop_slices: Optional[Tuple[slice, slice]] = None
    if crop:
        crop_slices = _crop_box_from_planes(images, margin=crop_margin)

    img_views: List[np.ndarray] = []
    for name in PLANES:
        img = images[name]
        if crop_slices is not None:
            img = img[crop_slices]
        img_views.append(img)

    img_norm = _global_log_norm(img_views, signed=False)

    fig, axes = plt.subplots(nrows=len(PLANES), ncols=1, figsize=(7, 15), constrained_layout=True)
    fig.suptitle(title)

    for i, name in enumerate(PLANES):
        img = img_views[i]
        vmax0 = _auto_vmax(img)
        ax0 = axes[i]
        im0 = ax0.imshow(
            img,
            origin="lower",
            vmax=None if img_norm is not None else vmax0,
            interpolation="nearest",
            norm=img_norm,
        )
        ax0.set_title(f"{name}: event (feature)")
        ax0.set_xlabel("x")
        ax0.set_ylabel("y")
        fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"[saved] {out_path}")
    else:
        plt.show()

    plt.close(fig)


# -------------------------
# Main
# -------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Plot a high-score event display and attribution overlay.")
    ap.add_argument("--shards_dir", default=getattr(cfg, "SHARDS_DIR", "shards"))
    ap.add_argument("--checkpoint", default=getattr(cfg, "CHECKPOINT_PATH", "checkpoints/checkpoint.pt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_samples", type=int, default=5000, help="How many random events to score before selecting.")
    ap.add_argument("--min_score", type=float, default=0.95, help="Score threshold for 'high-score' pool.")
    ap.add_argument("--topk_fallback", type=int, default=50, help="If no events exceed min_score, sample from top-K.")
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for event sampling. Defaults to None (random each run).",
    )
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--out", default="event_attrib.png", help="Output image path. Use --no_out to display.")
    ap.add_argument("--no_out", action="store_true", help="Show interactively instead of saving.")
    ap.add_argument("--channel", choices=["occ", "logq"], default="logq", help="Which input feature to attribute.")
    ap.add_argument("--signed", action="store_true", help="Keep signed attributions (default clips to positive).")
    ap.add_argument("--no_nnz_filter", action="store_true", help="Do not restrict sampling to nnz>0 events.")
    ap.add_argument("--crop", action="store_true", help="Crop plots to ROI around hits (recommended).")
    ap.add_argument("--crop_margin", type=int, default=10, help="Margin (pixels) when --crop is enabled.")
    ap.add_argument(
        "--signal_only",
        action="store_true",
        help="Sample only signal-labeled events and plot only the event display (no attributions).",
    )
    args = ap.parse_args(argv)

    shards_dir = str(args.shards_dir)
    ckpt = resolve_checkpoint(str(args.checkpoint))
    device = torch.device(str(args.device))

    # Load meta to get H/W that matches the shard data
    meta = torch.load(f"{shards_dir}/index.pt", map_location="cpu")
    h = int(meta.get("H", getattr(cfg, "H", 512)))
    w = int(meta.get("W", getattr(cfg, "W", 512)))

    # Build model (must match training config)
    embed_dim = int(getattr(cfg, "EMBED_DIM", 256))
    backbone = make_backbone(str(getattr(cfg, "BACKBONE", "small")), in_ch=2, embed_dim=embed_dim)
    model = MultiViewSetClassifier(backbone=backbone, embed_dim=embed_dim, plane_names=PLANES).to(device)

    state = torch.load(ckpt, map_location="cpu")
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(sd, strict=True)
    model.eval()

    # Pick a high-score event
    gi, score, meta_out = pick_random_high_score_event(
        model,
        shards_dir,
        num_samples=int(args.num_samples),
        min_score=float(args.min_score),
        topk_fallback=int(args.topk_fallback),
        seed=args.seed,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=device,
        restrict_to_nnz_gt0=not bool(args.no_nnz_filter),
        require_label=1 if bool(args.signal_only) else None,
    )

    print(
        f"[selected] global_event_id={gi}  score={score:.6f}  selection={meta_out['selection']}  true_label={meta_out['true_label']}"
    )

    # Load that event and compute attributions
    ds_one = ShardDataset(shards_dir, np.array([gi], dtype=np.int64), cache_size=1)
    coords, feats, _y = ds_one[0]

    channel = 0 if args.channel == "occ" else 1
    if args.signal_only:
        images = compute_event_images(
            coords,
            feats,
            h=h,
            w=w,
            channel=channel,
        )
        title = f"event {gi}  (selected={score:.4f})  true_label={meta_out['true_label']}"
        out = None if bool(args.no_out) else str(args.out)
        plot_event_only(
            images,
            out=out,
            title=title,
            crop=bool(args.crop),
            crop_margin=int(args.crop_margin),
        )
    else:
        images, attrs, score2 = compute_gradxinput_attribution(
            model,
            coords,
            feats,
            device=device,
            h=h,
            w=w,
            channel=channel,
            signed=bool(args.signed),
        )

        title = (
            f"event {gi}  model score={score2:.4f}  (selected={score:.4f})"
            f"  true_label={meta_out['true_label']}"
        )
        out = None if bool(args.no_out) else str(args.out)
        plot_event_and_attribution(
            images,
            attrs,
            out=out,
            title=title,
            crop=bool(args.crop),
            crop_margin=int(args.crop_margin),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
