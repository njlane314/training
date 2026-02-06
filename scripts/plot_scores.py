#!/usr/bin/env python3
"""
plot_scores.py

Compute the model score for *all* events in the shard sample (train/val/unused,
and optionally including nnz==0 events) and plot score distributions split by
signal vs background.

Default behavior:
  - Uses the latest checkpoint_step*.pt next to cfg.CHECKPOINT_PATH
  - Uses nominal event weights from index.pt
  - Includes all events (including nnz==0) unless --drop-empty is set
  - Plots probability score = sigmoid(logit) on [0,1] with density normalization

Examples:
  python plot_scores.py
  python plot_scores.py --out scores.png --norm none
  python plot_scores.py --drop-empty
  python plot_scores.py --score logit --xmin -12 --xmax 12
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

# Headless-safe plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import MinkowskiEngine as ME

# Support running either as a module inside a package or as a plain script
try:
    from . import config as cfg
    from .dataset import ShardDataset, collate_me_fusion
    from .fusion import MultiViewSetClassifier
    from .model import make_backbone
except Exception:
    import config as cfg
    from dataset import ShardDataset, collate_me_fusion
    from fusion import MultiViewSetClassifier
    from model import make_backbone


class EvalDataset(ShardDataset):
    """ShardDataset that also returns the nominal weight and the global event id."""
    def __getitem__(self, i: int):
        coords, feats, y = super().__getitem__(i)
        w = float(self.weights[i])
        gi = int(self.event_indices[i])
        return coords, feats, y, w, gi


def collate_me_fusion_with_w(batch):
    """
    Batch items: (coords, feats, y, w, gi)
    Returns: coords_by_plane, feats_by_plane, y, w, gi, available_mask
    """
    base = [(c, f, y) for (c, f, y, _, _) in batch]
    coords_by_plane, feats_by_plane, y, available_mask = collate_me_fusion(base, plane_names=("u", "v", "w"))
    w = torch.tensor([w for (_, _, _, w, _) in batch], dtype=torch.float32)
    gi = torch.tensor([gi for (_, _, _, _, gi) in batch], dtype=torch.int64)
    return coords_by_plane, feats_by_plane, y, w, gi, available_mask


def _find_latest_step_checkpoint(base_path: Path) -> Path:
    """
    Training writes checkpoints like:
      <stem>_step0000123<suffix>
    where base_path is cfg.CHECKPOINT_PATH.

    If base_path itself exists, use it.
    Otherwise, pick the highest-step *_step*.pt next to it.
    """
    if base_path.exists():
        return base_path

    parent = base_path.parent
    stem = base_path.stem
    suffix = base_path.suffix if base_path.suffix else ".pt"

    pat = re.compile(rf"^{re.escape(stem)}_step(\d+){re.escape(suffix)}$")
    best = None
    best_step = -1

    if parent.exists():
        for p in parent.iterdir():
            if not p.is_file():
                continue
            m = pat.match(p.name)
            if not m:
                continue
            step = int(m.group(1))
            if step > best_step:
                best_step = step
                best = p

    if best is None:
        raise FileNotFoundError(
            f"Could not find checkpoint:\n"
            f"  tried: {base_path}\n"
            f"  and:   {parent}/{stem}_step*{suffix}\n"
        )
    return best


def _build_model(device: torch.device, planes: Tuple[str, ...] = ("u", "v", "w")) -> torch.nn.Module:
    backbone = make_backbone(cfg.BACKBONE, in_ch=2, embed_dim=cfg.EMBED_DIM).to(device)
    model = MultiViewSetClassifier(backbone=backbone, embed_dim=cfg.EMBED_DIM, plane_names=planes).to(device)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-dir", default=cfg.SHARDS_DIR, help="Directory containing index.pt and shard_*.pt")
    ap.add_argument("--checkpoint", default=cfg.CHECKPOINT_PATH, help="Checkpoint path (or base path used by train.py)")
    ap.add_argument("--out", default="score_distribution.png", help="Output image path (png/pdf/etc)")
    ap.add_argument("--bins", type=int, default=50, help="Number of histogram bins")
    ap.add_argument("--batch-size", type=int, default=64, help="Eval batch size")
    ap.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 is safest for shard IO)")
    ap.add_argument("--cache-size", type=int, default=2, help="Shard cache size inside dataset")
    ap.add_argument("--drop-empty", action="store_true", help="Drop events with nnz==0 if index.pt provides nnz")
    ap.add_argument("--unweighted", action="store_true", help="Ignore nominal weights (use weight=1 for all events)")
    ap.add_argument(
        "--norm",
        choices=["density", "none"],
        default="density",
        help="Histogram normalization: density => unit area per class; none => counts/sumw",
    )
    ap.add_argument(
        "--score",
        choices=["prob", "logit"],
        default="prob",
        help="Score to histogram: prob=sigmoid(logit) or logit=raw output",
    )
    ap.add_argument("--xmin", type=float, default=None, help="Histogram x-min (optional)")
    ap.add_argument("--xmax", type=float, default=None, help="Histogram x-max (optional)")
    ap.add_argument("--logy", action="store_true", help="Use log scale on y-axis")
    ap.add_argument("--save-npz", default="", help="Optional .npz output storing per-event arrays (scores, y, w, gi)")
    args = ap.parse_args()

    shards_dir = Path(args.shards_dir)
    meta_path = shards_dir / "index.pt"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")

    meta = torch.load(meta_path, map_location="cpu")
    n_events = int(meta["n_events"])
    labels_all = np.asarray(meta["labels"], dtype=np.uint8).reshape(-1)
    weights_all = np.asarray(meta["weights"], dtype=np.float32).reshape(-1)

    if labels_all.shape[0] != n_events or weights_all.shape[0] != n_events:
        raise RuntimeError(
            f"index.pt mismatch: n_events={n_events}, labels={labels_all.shape[0]}, weights={weights_all.shape[0]}"
        )

    if args.drop_empty and ("nnz" in meta) and (meta["nnz"] is not None):
        nnz = meta["nnz"]
        if isinstance(nnz, torch.Tensor):
            nnz_all = nnz.to(dtype=torch.int64).cpu().numpy().reshape(-1)
        else:
            nnz_all = np.asarray(nnz, dtype=np.int64).reshape(-1)
        if nnz_all.shape[0] != n_events:
            raise RuntimeError(f"index.pt nnz mismatch: nnz={nnz_all.shape[0]} vs n_events={n_events}")

        keep = nnz_all > 0
        event_indices = np.flatnonzero(keep).astype(np.int64, copy=False)
        dropped = int(n_events - event_indices.size)
        print(f"[data] drop-empty enabled: keeping {event_indices.size}/{n_events} (dropped {dropped})")
    else:
        event_indices = np.arange(n_events, dtype=np.int64)
        if ("nnz" in meta) and (meta["nnz"] is not None):
            nnz = meta["nnz"]
            if isinstance(nnz, torch.Tensor):
                nnz_all = nnz.to(dtype=torch.int64).cpu().numpy().reshape(-1)
            else:
                nnz_all = np.asarray(nnz, dtype=np.int64).reshape(-1)
            if nnz_all.shape[0] == n_events:
                n_empty = int((nnz_all == 0).sum())
                print(f"[data] including all events: {n_events} total (nnz==0: {n_empty})")
            else:
                print(f"[data] including all events: {n_events} total")
        else:
            print(f"[data] including all events: {n_events} total")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # Model + checkpoint
    ckpt_base = Path(args.checkpoint)
    ckpt_path = _find_latest_step_checkpoint(ckpt_base)
    print(f"[ckpt] using {ckpt_path}")

    model = _build_model(device=device, planes=("u", "v", "w"))
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[ckpt] missing keys: {len(missing)}")
    if unexpected:
        print(f"[ckpt] unexpected keys: {len(unexpected)}")
    model.eval()

    # Dataset / loader
    ds = EvalDataset(str(shards_dir), event_indices, cache_size=args.cache_size)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate_me_fusion_with_w,
        pin_memory=True,
        persistent_workers=(int(args.num_workers) > 0),
    )

    # Histogram setup
    bins = int(args.bins)
    if bins <= 0:
        raise ValueError("--bins must be > 0")

    if args.xmin is None or args.xmax is None:
        if args.score == "prob":
            xmin = 0.0 if args.xmin is None else float(args.xmin)
            xmax = 1.0 if args.xmax is None else float(args.xmax)
        else:
            xmin = -10.0 if args.xmin is None else float(args.xmin)
            xmax = 10.0 if args.xmax is None else float(args.xmax)
    else:
        xmin, xmax = float(args.xmin), float(args.xmax)

    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        raise ValueError(f"Invalid histogram range: xmin={xmin}, xmax={xmax}")

    edges = np.linspace(xmin, xmax, bins + 1, dtype=np.float64)
    bin_w = float(edges[1] - edges[0])

    hs = np.zeros(bins, dtype=np.float64)
    hb = np.zeros(bins, dtype=np.float64)

    n_sig = 0
    n_bkg = 0
    sumw_sig = 0.0
    sumw_bkg = 0.0

    # Optional per-event outputs
    save_arrays = bool(args.save_npz)
    scores_out = []
    y_out = []
    w_out = []
    gi_out = []

    with torch.no_grad():
        for coords_by_plane, feats_by_plane, y, w, gi, available_mask in dl:
            # Score weights
            if args.unweighted:
                w_use = torch.ones_like(w)
            else:
                w_use = w

            # Build ME sparse inputs (coords CPU, feats on device)
            inputs: Dict[str, ME.SparseTensor] = {}
            for name in ("u", "v", "w"):
                feats = feats_by_plane[name].to(device, non_blocking=True)
                coords = coords_by_plane[name]  # CPU int32
                inputs[name] = ME.SparseTensor(
                    features=feats,
                    coordinates=coords,
                    device=device,
                )

            logits = model(inputs, available_mask=available_mask.to(device, non_blocking=True)).squeeze(1)

            if args.score == "prob":
                score = torch.sigmoid(logits)
            else:
                score = logits

            score_np = score.detach().cpu().numpy().astype(np.float64, copy=False)
            y_np = y.detach().cpu().numpy().astype(np.float64, copy=False)
            w_np = w_use.detach().cpu().numpy().astype(np.float64, copy=False)
            gi_np = gi.detach().cpu().numpy().astype(np.int64, copy=False)

            sig = y_np > 0.5
            bkg = ~sig

            if sig.any():
                hs += np.histogram(score_np[sig], bins=edges, weights=w_np[sig])[0]
                n_sig += int(sig.sum())
                sumw_sig += float(w_np[sig].sum())

            if bkg.any():
                hb += np.histogram(score_np[bkg], bins=edges, weights=w_np[bkg])[0]
                n_bkg += int(bkg.sum())
                sumw_bkg += float(w_np[bkg].sum())

            if save_arrays:
                scores_out.append(score_np)
                y_out.append(y_np)
                w_out.append(w_np)
                gi_out.append(gi_np)

    if args.norm == "density":
        # Normalize each class to unit area: sum(hist * bin_width) == 1
        if hs.sum() > 0:
            hs = hs / (hs.sum() * bin_w)
        if hb.sum() > 0:
            hb = hb / (hb.sum() * bin_w)

    centers = 0.5 * (edges[:-1] + edges[1:])

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.step(centers, hb, where="mid", label=f"Background (N={n_bkg}, sumw={sumw_bkg:.3g})")
    ax.step(centers, hs, where="mid", label=f"Signal (N={n_sig}, sumw={sumw_sig:.3g})")

    if args.score == "prob":
        ax.set_xlabel("Score = sigmoid(logit)")
    else:
        ax.set_xlabel("Score = logit")

    if args.norm == "density":
        ax.set_ylabel("Density")
    else:
        ax.set_ylabel("Events" if args.unweighted else "Sum of weights")

    if args.logy:
        ax.set_yscale("log")

    ax.set_xlim(xmin, xmax)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[out] wrote {out_path}")

    if save_arrays:
        scores_all = np.concatenate(scores_out, axis=0) if scores_out else np.zeros((0,), dtype=np.float64)
        y_all = np.concatenate(y_out, axis=0) if y_out else np.zeros((0,), dtype=np.float64)
        w_all = np.concatenate(w_out, axis=0) if w_out else np.zeros((0,), dtype=np.float64)
        gi_all = np.concatenate(gi_out, axis=0) if gi_out else np.zeros((0,), dtype=np.int64)

        npz_path = Path(args.save_npz)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            npz_path,
            score=scores_all,
            y=y_all,
            w=w_all,
            event_index=gi_all,
            score_type=args.score,
        )
        print(f"[out] wrote {npz_path} (arrays: score,y,w,event_index)")


if __name__ == "__main__":
    main()
