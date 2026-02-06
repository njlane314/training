#!/usr/bin/env python3
"""
llr_consistency_plot.py

Make the LLR self-consistency plot from checkpoints produced by your train.py.

For each selected checkpoint:
  - run inference on a fixed held-out validation split (constructed deterministically
    from SHARDS_DIR/index.pt using SEED and VAL_FRACTION, matching your train.py split logic)
  - collect logits z, labels y, and nominal weights w_nominal
  - estimate (weighted) class-conditional 1D distributions in score space:
        p_s(z), p_b(z)
    via weighted histograms
  - compute
        r(z) = log p_s(z) - log p_b(z)
  - plot r(z) vs z and overlay the identity line y=x

Outputs (in --out-dir):
  - llr_consistency.png
  - metrics.csv  (per-checkpoint: slope/intercept of r vs z, MSE of r-z, etc.)

Usage:
  python llr_consistency_plot.py --ckpt-base checkpoints/ckpt.pt --out-dir llr_diag
  python llr_consistency_plot.py --ckpt-glob "checkpoints/ckpt_step*.pt" --plot-count 8

Notes:
  - This script does NO training.
  - It assumes your shards exist (SHARDS_DIR/index.pt + shard_*.pt).
  - It assumes checkpoints are saved as: <stem>_stepXXXXXXX<suffix>
    (exactly as in your train.py _checkpoint_path_for_step()).
"""

from __future__ import annotations

import argparse
import array
import csv
import glob
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


PLANES = ("u", "v", "w")


def _load_meta(shards_dir: str) -> dict:
    return torch.load(f"{shards_dir}/index.pt", map_location="cpu")


def _compute_splits_from_meta(
    meta: dict,
    *,
    seed: int,
    val_fraction: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Matches your train.py split logic:
      - optionally filter nnz>0 (if present in index.pt)
      - rng permute
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


def _sort_event_indices_for_io(meta: dict, event_idx: np.ndarray) -> np.ndarray:
    """
    Sort by (shard_id, local_id) to reduce shard thrash during sequential inference.
    """
    shard_events = int(meta.get("shard_events", getattr(cfg, "SHARD_EVENTS", 2048)))
    shard_id = (event_idx // shard_events).astype(np.int64, copy=False)
    local_id = (event_idx - shard_id * shard_events).astype(np.int64, copy=False)
    key = np.lexsort((local_id, shard_id))
    return event_idx[key].astype(np.int64, copy=False)


class ShardDatasetWithWeights(ShardDataset):
    """
    Same as ShardDataset, but returns nominal weight too.
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


@torch.no_grad()
def _infer_scores(
    model: nn.Module,
    dl_val: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    max_batches: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns arrays (z, y, w):
      z = logits (float64)
      y = labels (uint8, 0/1)
      w = nominal weights (float64, clipped to >=0 for consistency with training sampler)
    """
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
        ys.append(y.detach().cpu().numpy().astype(np.uint8, copy=False))
        ws.append(w.detach().cpu().numpy().astype(np.float64, copy=False))

    z = np.concatenate(zs, axis=0) if zs else np.zeros((0,), dtype=np.float64)
    y = np.concatenate(ys, axis=0) if ys else np.zeros((0,), dtype=np.uint8)
    w = np.concatenate(ws, axis=0) if ws else np.zeros((0,), dtype=np.float64)
    w = np.clip(w, 0.0, None)  # match training sampler behavior
    return z, y, w


@dataclass
class CurveMetrics:
    step: int
    n_val: int
    nbins: int
    q_lo: float
    q_hi: float
    z_lo: float
    z_hi: float
    n_bins_used: int
    alpha: float
    beta: float
    mse_r_minus_z: float


def _llr_curve_from_scores(
    z: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    *,
    step: int,
    nbins: int,
    q_lo: float,
    q_hi: float,
) -> Tuple[np.ndarray, np.ndarray, CurveMetrics]:
    """
    Build r(z)=log p_s(z)/p_b(z) using weighted histograms in score space.

    Returns:
      x: bin centers where both classes have support
      r: log-ratio at those centers
      metrics: slope/intercept fit and MSE of (r - x)
    """
    if z.size == 0:
        raise ValueError("empty score array (no validation events processed?)")

    z = z.astype(np.float64, copy=False)
    y = y.astype(np.uint8, copy=False)
    w = np.clip(w.astype(np.float64, copy=False), 0.0, None)

    z_lo = float(np.quantile(z, float(q_lo)))
    z_hi = float(np.quantile(z, float(q_hi)))
    if not np.isfinite(z_lo) or not np.isfinite(z_hi) or z_hi <= z_lo:
        m = float(np.mean(z))
        s = float(np.std(z) + 1e-6)
        z_lo, z_hi = m - 4.0 * s, m + 4.0 * s

    edges = np.linspace(z_lo, z_hi, int(nbins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    ys = (y == 1)
    yb = (y == 0)
    if ys.sum() == 0 or yb.sum() == 0:
        raise ValueError("need both signal and background in the evaluation sample")

    ws = w[ys]
    wb = w[yb]
    if ws.sum() <= 0 or wb.sum() <= 0:
        raise ValueError("weights must sum to >0 within each class (after clipping)")

    hs, _ = np.histogram(z[ys], bins=edges, weights=ws)
    hb, _ = np.histogram(z[yb], bins=edges, weights=wb)

    ps = hs / max(hs.sum(), 1e-300)
    pb = hb / max(hb.sum(), 1e-300)

    mask = (ps > 0) & (pb > 0) & np.isfinite(ps) & np.isfinite(pb)
    x = centers[mask]
    r = (np.log(ps[mask]) - np.log(pb[mask]))

    if x.size >= 2:
        alpha, beta = np.polyfit(x, r, deg=1)
        mse = float(np.mean((r - x) ** 2))
    else:
        alpha, beta, mse = float("nan"), float("nan"), float("nan")

    metrics = CurveMetrics(
        step=int(step),
        n_val=int(z.size),
        nbins=int(nbins),
        q_lo=float(q_lo),
        q_hi=float(q_hi),
        z_lo=float(z_lo),
        z_hi=float(z_hi),
        n_bins_used=int(x.size),
        alpha=float(alpha),
        beta=float(beta),
        mse_r_minus_z=float(mse),
    )
    return x, r, metrics


def _to_root_graph(
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: int,
    name: str,
):
    """
    Make a ROOT.TGraph with consistent styling (line+markers),
    similar to plot_loss_root.py.
    """
    import ROOT  # type: ignore

    # ROOT wants Python array('d') or buffers of doubles.
    xs = array.array("d", [float(v) for v in np.asarray(x, dtype=np.float64).reshape(-1)])
    ys = array.array("d", [float(v) for v in np.asarray(y, dtype=np.float64).reshape(-1)])
    gr = ROOT.TGraph(len(xs), xs, ys)
    gr.SetName(name)

    gr.SetLineColor(color)
    gr.SetMarkerColor(color)
    gr.SetLineWidth(1)
    gr.SetMarkerStyle(20)
    gr.SetMarkerSize(0.45)
    return gr


def _checkpoint_glob_from_base(ckpt_base: Path) -> str:
    # train.py creates: <stem>_stepXXXXXXX<suffix>
    stem = ckpt_base.stem
    suffix = ckpt_base.suffix
    return str(ckpt_base.with_name(f"{stem}_step*{suffix}"))


def _list_checkpoints(glob_pat: str) -> List[Path]:
    paths = [Path(p) for p in glob.glob(glob_pat)]
    if not paths:
        raise FileNotFoundError(f"no checkpoints matched glob: {glob_pat}")

    def step_of(p: Path) -> int:
        # Prefer parsing from filename; fallback to reading file.
        m = re.search(r"_step(\d+)\.", p.name)
        if m:
            return int(m.group(1))
        ck = torch.load(p, map_location="cpu")
        return int(ck.get("step", -1))

    paths.sort(key=step_of)
    return paths


def _select_checkpoints(
    ckpts: List[Path],
    *,
    plot_count: int,
    steps: Optional[List[int]],
) -> List[Path]:
    if steps is not None and len(steps) > 0:
        step_to_path: Dict[int, Path] = {}
        for p in ckpts:
            ck = torch.load(p, map_location="cpu")
            step_to_path[int(ck.get("step", -1))] = p

        out: List[Path] = []
        missing: List[int] = []
        for s in steps:
            if int(s) in step_to_path:
                out.append(step_to_path[int(s)])
            else:
                missing.append(int(s))
        if missing:
            avail = sorted(step_to_path.keys())
            raise FileNotFoundError(
                f"requested steps not found: {missing}. available steps (first 30): {avail[:30]} ..."
            )
        return out

    # Evenly spaced selection across the list.
    if plot_count <= 0 or len(ckpts) == 0:
        return []
    if len(ckpts) <= plot_count:
        return ckpts
    idxs = np.linspace(0, len(ckpts) - 1, plot_count, dtype=int)
    return [ckpts[int(i)] for i in idxs]


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--ckpt-base",
        type=str,
        default=getattr(cfg, "CHECKPOINT_PATH", None),
        help="Same path you used for cfg.CHECKPOINT_PATH in training (e.g. checkpoints/ckpt.pt). "
             "This script will glob <stem>_step*<suffix> next to it.",
    )
    ap.add_argument(
        "--ckpt-glob",
        type=str,
        default=None,
        help='Explicit glob for checkpoints, e.g. "checkpoints/ckpt_step*.pt". '
             "Overrides --ckpt-base if set.",
    )
    ap.add_argument("--out-dir", type=str, default="llr_diag")
    ap.add_argument("--plot-name", type=str, default="llr_consistency.png")

    ap.add_argument("--nbins", type=int, default=60)
    ap.add_argument("--q-lo", type=float, default=0.005)
    ap.add_argument("--q-hi", type=float, default=0.995)

    ap.add_argument(
        "--plot-count",
        type=int,
        default=6,
        help="How many checkpoints to overlay (evenly spaced). Ignored if --steps is set.",
    )
    ap.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated list of checkpoint steps to plot (exact match), e.g. '0,2000,4000'.",
    )
    ap.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="If set, cap how many validation batches to use per checkpoint (for speed).",
    )
    ap.add_argument(
        "--val-size",
        type=int,
        default=None,
        help="If set, use only the first N events of the (deterministic) val split (then IO-sorted).",
    )
    ap.add_argument("--batch-size", type=int, default=int(cfg.BATCH_SIZE))
    ap.add_argument("--num-workers", type=int, default=int(cfg.NUM_WORKERS))

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.ckpt_glob is not None:
        glob_pat = args.ckpt_glob
    else:
        if args.ckpt_base is None:
            raise ValueError("Need --ckpt-glob or --ckpt-base (and cfg.CHECKPOINT_PATH was not set).")
        glob_pat = _checkpoint_glob_from_base(Path(args.ckpt_base))

    ckpts = _list_checkpoints(glob_pat)

    steps_list: Optional[List[int]] = None
    if args.steps is not None:
        steps_list = [int(s.strip()) for s in args.steps.split(",") if s.strip()]

    selected = _select_checkpoints(ckpts, plot_count=int(args.plot_count), steps=steps_list)
    if not selected:
        raise RuntimeError("No checkpoints selected for plotting.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta = _load_meta(cfg.SHARDS_DIR)
    _, val_idx = _compute_splits_from_meta(meta, seed=int(cfg.SEED), val_fraction=float(cfg.VAL_FRACTION))

    if args.val_size is not None:
        n = int(args.val_size)
        if n <= 0:
            raise ValueError("--val-size must be > 0")
        val_idx = val_idx[: min(n, val_idx.size)]

    val_idx = _sort_event_indices_for_io(meta, val_idx)

    ds_val = ShardDatasetWithWeights(cfg.SHARDS_DIR, val_idx, cache_size=2)
    dl_val = torch.utils.data.DataLoader(
        ds_val,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate_me_fusion_with_weights,
        pin_memory=True,
        persistent_workers=(int(args.num_workers) > 0),
    )

    model = _build_model_from_cfg(device)

    curves: List[Tuple[int, np.ndarray, np.ndarray]] = []
    metrics_all: List[CurveMetrics] = []

    for p in selected:
        ck = torch.load(p, map_location="cpu")
        step = int(ck.get("step", -1))
        state = ck.get("model", None)
        if state is None:
            raise KeyError(f"checkpoint {p} does not contain key 'model'")

        model.load_state_dict(state, strict=True)

        z, y, w = _infer_scores(model, dl_val, device, max_batches=args.max_val_batches)
        x, r, m = _llr_curve_from_scores(
            z=z,
            y=y,
            w=w,
            step=step,
            nbins=int(args.nbins),
            q_lo=float(args.q_lo),
            q_hi=float(args.q_hi),
        )

        curves.append((step, x, r))
        metrics_all.append(m)

        print(
            f"[plot] step={step:7d}  n_val={m.n_val:6d}  "
            f"bins_used={m.n_bins_used:3d}  z_range=[{m.z_lo:.2f},{m.z_hi:.2f}]  "
            f"alpha={m.alpha:.3f} beta={m.beta:.3f} mse={m.mse_r_minus_z:.3f}"
        )

    # Write metrics.csv
    metrics_path = out_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        wcsv = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "n_val",
                "nbins",
                "q_lo",
                "q_hi",
                "z_lo",
                "z_hi",
                "n_bins_used",
                "alpha",
                "beta",
                "mse_r_minus_z",
            ],
        )
        wcsv.writeheader()
        for m in sorted(metrics_all, key=lambda t: t.step):
            wcsv.writerow(vars(m))

    # Plot (PyROOT)
    curves.sort(key=lambda t: t[0])
    plot_path = out_dir / str(args.plot_name)

    try:
        import ROOT  # type: ignore
    except ImportError:
        sys.stderr.write(
            "ERROR: PyROOT is not available (this script expects `import ROOT` to work).\n"
        )
        raise SystemExit(2)

    ROOT.gROOT.SetBatch(True)

    # Style knobs to resemble plot_loss_root.py
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(0)
    ROOT.gStyle.SetPadTickX(1)
    ROOT.gStyle.SetPadTickY(1)
    ROOT.gStyle.SetLegendBorderSize(0)

    good = [(step, x, r) for (step, x, r) in curves if x is not None and r is not None and x.size >= 2]
    if not good:
        raise RuntimeError("No curves have >=2 points; nothing to plot.")

    allx = np.concatenate([x for (_, x, _) in good], axis=0)
    allr = np.concatenate([r for (_, _, r) in good], axis=0)

    xmin_raw = float(np.min(allx))
    xmax_raw = float(np.max(allx))
    ymin_raw = float(np.min(np.concatenate([allr, allx], axis=0)))  # include y=x reference
    ymax_raw = float(np.max(np.concatenate([allr, allx], axis=0)))

    # Linear margins (avoid zero range)
    dx = max(xmax_raw - xmin_raw, 1e-6)
    dy = max(ymax_raw - ymin_raw, 1e-6)
    xmin = xmin_raw - 0.05 * dx
    xmax = xmax_raw + 0.05 * dx
    ymin = ymin_raw - 0.05 * dy
    ymax = ymax_raw + 0.05 * dy

    c = ROOT.TCanvas("c_llr", "c_llr", 900, 650)
    c.SetLeftMargin(0.12)
    c.SetRightMargin(0.12)
    c.SetBottomMargin(0.12)
    c.SetTopMargin(0.06)

    # Color cycle for multiple checkpoints
    colors = [
        ROOT.kBlue + 1,
        ROOT.kRed + 1,
        ROOT.kGreen + 2,
        ROOT.kMagenta + 1,
        ROOT.kOrange + 7,
        ROOT.kCyan + 1,
        ROOT.kViolet + 1,
        ROOT.kBlack,
    ]

    graphs: List[Tuple[int, "ROOT.TGraph"]] = []
    for i, (step, x, r) in enumerate(good):
        col = int(colors[i % len(colors)])
        gr = _to_root_graph(x, r, color=col, name=f"gr_step{int(step)}")
        graphs.append((int(step), gr))

    if not graphs:
        raise RuntimeError("No ROOT graphs created; nothing to plot.")

    # Base graph drives axes/ranges
    base = graphs[0][1]
    base.SetMinimum(ymin)
    base.SetMaximum(ymax)
    base.Draw("ALP")
    base.GetXaxis().SetLimits(xmin, xmax)
    base.GetXaxis().SetTitle("z (model logit)")
    base.GetYaxis().SetTitle("r(z) = log p_{s}(z) - log p_{b}(z)")
    base.GetYaxis().SetTitleOffset(1.2)
    base.GetXaxis().SetTitleOffset(1.0)

    base.GetXaxis().SetTitleSize(0.05)
    base.GetYaxis().SetTitleSize(0.05)
    base.GetXaxis().SetLabelSize(0.04)
    base.GetYaxis().SetLabelSize(0.04)

    # Identity line y=x
    line = ROOT.TLine(xmin, xmin, xmax, xmax)
    line.SetLineStyle(2)
    line.SetLineWidth(1)
    line.Draw("same")

    # Overlay remaining checkpoints
    for _, gr in graphs[1:]:
        gr.Draw("LP same")

    # Legend (auto-sized to number of entries)
    n_entries = 1 + len(graphs)  # identity + curves
    top = 0.88
    right = 0.88
    left = 0.58
    height = 0.05 * float(n_entries)
    bottom = max(0.18, top - height)
    leg = ROOT.TLegend(left, bottom, right, top)
    leg.SetFillStyle(0)
    leg.SetBorderSize(0)
    leg.AddEntry(line, "y = x", "l")
    for step, gr in graphs:
        leg.AddEntry(gr, f"step {step}", "lp")
    leg.Draw()

    c.Modified()
    c.Update()
    c.SaveAs(str(plot_path))

    print(f"[done] wrote {plot_path} and {metrics_path}")


if __name__ == "__main__":
    main()
