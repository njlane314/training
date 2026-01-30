#!/usr/bin/env python3
"""
plot_random_sig_bkg.py

Pick one random signal (label==1) and one random background (label==0) event and
plot the U/V/W planes as dense images.

Two sources:
  --source shards : uses sparse shards (index.pt + shard_*.pt) and reconstructs dense planes
  --source root   : reads the ROOT file directly and applies the same transform as your sparsifier

Examples
--------
# From shards (recommended if you train from shards):
python plot_random_sig_bkg.py --source shards --shards-dir /path/to/sparse_shards --out rand_shards.png

# From ROOT (shows "what the sparsifier would keep" by default):
python plot_random_sig_bkg.py --source root --root-file /gluster/data/dune/niclane/events.root --tree events --out rand_root.png

# Force a specific event index:
python plot_random_sig_bkg.py --source shards --event-sig 123 --event-bkg 456
"""

import argparse
import os
import sys
import math

import numpy as np
import torch


def _load_index(shards_dir: str):
    idx_path = os.path.join(shards_dir, "index.pt")
    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"missing index.pt at: {idx_path}")
    meta = torch.load(idx_path, map_location="cpu")
    labels = np.asarray(meta["labels"], dtype=np.uint8).reshape(-1)
    shard_events = int(meta["shard_events"])
    H = int(meta.get("H", 512))
    W = int(meta.get("W", 512))
    nnz = np.asarray(meta.get("nnz", np.zeros_like(labels, dtype=np.int32)), dtype=np.int32).reshape(-1)
    branches = meta.get("branches", {})
    return meta, labels, nnz, shard_events, H, W, branches


def _pick_event(labels: np.ndarray, rng: np.random.Generator, cls: int, event_override=None, skip_placeholder=False, nnz=None):
    if event_override is not None:
        gi = int(event_override)
        if gi < 0 or gi >= labels.shape[0]:
            raise ValueError(f"event index out of range: {gi} (n_events={labels.shape[0]})")
        if int(labels[gi]) != int(cls):
            raise ValueError(f"event {gi} has label={int(labels[gi])}, expected {cls}")
        if skip_placeholder and nnz is not None and int(nnz[gi]) == 1:
            raise ValueError(f"event {gi} is placeholder (nnz==1) and --skip-placeholder was set")
        return gi

    idx = np.where(labels == cls)[0]
    if idx.size == 0:
        raise RuntimeError(f"no events with label {cls}")

    # Optionally avoid placeholder events (nnz==1)
    if skip_placeholder and nnz is not None:
        idx2 = idx[nnz[idx] != 1]
        if idx2.size > 0:
            idx = idx2

    return int(rng.choice(idx, size=1, replace=False)[0])


def _sparse_event_from_shards(shards_dir: str, gi: int, shard_events: int):
    sid = gi // shard_events
    shard_path = os.path.join(shards_dir, f"shard_{sid:05d}.pt")
    if not os.path.exists(shard_path):
        raise FileNotFoundError(f"missing shard file: {shard_path}")

    d = torch.load(shard_path, map_location="cpu")
    start_event = int(d["start_event"])
    local = gi - start_event
    if local < 0 or local + 1 >= int(d["starts"].numel()):
        raise RuntimeError(f"event {gi} not found in shard {sid:05d} (start_event={start_event})")

    s = int(d["starts"][local].item())
    e = int(d["starts"][local + 1].item())
    coords = d["coords"][s:e].cpu().numpy().astype(np.int64, copy=False)  # [nnz,3] = [view,y,x]
    feats = d["feats"][s:e].cpu().numpy().astype(np.float32, copy=False)  # [nnz,4], feats[:,0]=adc
    return coords, feats


def _sparse_to_dense_planes(coords: np.ndarray, feats: np.ndarray, H: int, W: int, n_views: int = 3):
    planes = [np.zeros((H, W), dtype=np.float32) for _ in range(n_views)]
    if coords.size == 0:
        return planes

    view = coords[:, 0].astype(np.int64, copy=False)
    yy = coords[:, 1].astype(np.int64, copy=False)
    xx = coords[:, 2].astype(np.int64, copy=False)
    adc = feats[:, 0].astype(np.float32, copy=False)

    for v in range(n_views):
        m = (view == v)
        if not np.any(m):
            continue
        yv = yy[m]
        xv = xx[m]
        av = adc[m]
        # indices should be unique per (view,y,x) in your construction; assignment is fine
        planes[v][yv, xv] = av
    return planes


def _plane_transform_like_sparsifier(flat: np.ndarray, H: int, W: int, thr: float, signlog: bool):
    flat = np.asarray(flat).reshape(-1)
    if flat.size != H * W:
        raise ValueError(f"plane size {flat.size} != {H}*{W}")

    out = np.zeros((H * W,), dtype=np.float32)

    if signlog:
        if thr <= 0.0:
            idx = np.flatnonzero(flat)
        else:
            idx = np.flatnonzero(np.abs(flat) > thr)
        if idx.size:
            val = flat[idx].astype(np.float32, copy=False)
            out[idx] = np.sign(val) * np.log1p(np.abs(val))
    else:
        idx = np.flatnonzero(flat > thr)
        if idx.size:
            val = flat[idx].astype(np.float32, copy=False)
            out[idx] = np.log1p(np.maximum(val, 0.0))

    return out.reshape(H, W)


def _event_from_root(root_file: str, tree: str, gi: int, H: int, W: int,
                    br_y: str, br_u: str, br_v: str, br_w: str,
                    thr: float, signlog: bool):
    try:
        import uproot
    except Exception as e:
        raise RuntimeError("uproot is required for --source root") from e

    with uproot.open(root_file) as f:
        t = f[tree]

        # load label for sanity
        y_arr = t[br_y].array(entry_start=gi, entry_stop=gi + 1, library="np")
        y_val = int(np.asarray(y_arr).reshape(-1)[0])

        a = t.arrays([br_u, br_v, br_w], entry_start=gi, entry_stop=gi + 1, library="np")
        u0 = np.asarray(a[br_u][0]).reshape(-1)
        v0 = np.asarray(a[br_v][0]).reshape(-1)
        w0 = np.asarray(a[br_w][0]).reshape(-1)

        U = _plane_transform_like_sparsifier(u0, H, W, thr, signlog)
        V = _plane_transform_like_sparsifier(v0, H, W, thr, signlog)
        Wp = _plane_transform_like_sparsifier(w0, H, W, thr, signlog)

    return y_val, [U, V, Wp]


def _robust_vmin_vmax(planes_list):
    # planes_list: list of 2 events * 3 planes each, already dense
    vals = np.concatenate([p.reshape(-1) for p in planes_list], axis=0)
    nz = vals[vals != 0]
    if nz.size == 0:
        return -1.0, 1.0

    lo = float(np.percentile(nz, 1))
    hi = float(np.percentile(nz, 99))

    if lo < 0 < hi:
        m = max(abs(lo), abs(hi))
        return -m, m
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["shards", "root"], default="shards",
                    help="where to load events from")
    ap.add_argument("--shards-dir", default=os.environ.get("SHARDS_DIR", ""),
                    help="directory containing index.pt and shard_*.pt")
    ap.add_argument("--root-file", default=os.environ.get("ROOT_FILE", ""),
                    help="ROOT file path (needed for --source root)")
    ap.add_argument("--tree", default=os.environ.get("TREE", "events"),
                    help="ROOT tree name (for --source root)")
    ap.add_argument("--h", type=int, default=None, help="image height (default: from index.pt or 512)")
    ap.add_argument("--w", type=int, default=None, help="image width (default: from index.pt or 512)")

    # Branch names (only used for --source root; for shards we only need index.pt/shards)
    ap.add_argument("--br-y", default=os.environ.get("BR_Y", "is_signal"))
    ap.add_argument("--br-u", default=os.environ.get("BR_U", "detector_image_u"))
    ap.add_argument("--br-v", default=os.environ.get("BR_V", "detector_image_v"))
    ap.add_argument("--br-w", default=os.environ.get("BR_W", "detector_image_w"))

    # Transform settings for ROOT-mode visualization (match your sparsifier)
    ap.add_argument("--thr", type=float, default=float(os.environ.get("THRESH", "0.0")))
    ap.add_argument("--signlog", action="store_true", default=(os.environ.get("ADC_SIGNLOG", "0") != "0"))

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--event-sig", type=int, default=None, help="global event index for signal")
    ap.add_argument("--event-bkg", type=int, default=None, help="global event index for background")
    ap.add_argument("--skip-placeholder", action="store_true",
                    help="avoid nnz==1 placeholder events (requires index.pt nnz)")
    ap.add_argument("--out", default="random_sig_bkg.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load meta if available (needed for shards mode; also helpful for H/W in root mode)
    meta = labels = nnz = shard_events = H = W = branches = None
    if args.shards_dir:
        try:
            meta, labels, nnz, shard_events, H0, W0, branches = _load_index(args.shards_dir)
            H = H0
            W = W0
        except FileNotFoundError:
            if args.source == "shards":
                raise
            # root mode can proceed without shards meta

    if args.h is not None:
        H = int(args.h)
    if args.w is not None:
        W = int(args.w)
    if H is None:
        H = 512
    if W is None:
        W = 512

    if args.source == "shards":
        if not args.shards_dir:
            raise ValueError("--shards-dir is required for --source shards")
        if labels is None or shard_events is None:
            raise RuntimeError("failed to load index.pt from --shards-dir")

        gi_sig = _pick_event(labels, rng, cls=1, event_override=args.event_sig,
                             skip_placeholder=args.skip_placeholder, nnz=nnz)
        gi_bkg = _pick_event(labels, rng, cls=0, event_override=args.event_bkg,
                             skip_placeholder=args.skip_placeholder, nnz=nnz)

        cS, fS = _sparse_event_from_shards(args.shards_dir, gi_sig, shard_events)
        cB, fB = _sparse_event_from_shards(args.shards_dir, gi_bkg, shard_events)

        planes_sig = _sparse_to_dense_planes(cS, fS, H, W)
        planes_bkg = _sparse_to_dense_planes(cB, fB, H, W)

        # Print quick diagnostics
        print(f"[shards] signal gi={gi_sig} nnz={cS.shape[0]} adc(min/mean/max)={float(fS[:,0].min()):+.3e}/{float(fS[:,0].mean()):+.3e}/{float(fS[:,0].max()):+.3e}")
        print(f"[shards] bkg    gi={gi_bkg} nnz={cB.shape[0]} adc(min/mean/max)={float(fB[:,0].min()):+.3e}/{float(fB[:,0].mean()):+.3e}/{float(fB[:,0].max()):+.3e}")
        title_sig = f"signal (label=1) gi={gi_sig} nnz={cS.shape[0]}"
        title_bkg = f"bkg (label=0) gi={gi_bkg} nnz={cB.shape[0]}"

    else:
        if not args.root_file:
            raise ValueError("--root-file is required for --source root")

        # If we have index labels, pick by that; otherwise read labels from ROOT
        if labels is None:
            import uproot
            with uproot.open(args.root_file) as f:
                t = f[args.tree]
                labels = np.asarray(t[args.br_y].array(library="np"), dtype=np.uint8).reshape(-1)
                nnz = None

        gi_sig = _pick_event(labels, rng, cls=1, event_override=args.event_sig,
                             skip_placeholder=False, nnz=None)
        gi_bkg = _pick_event(labels, rng, cls=0, event_override=args.event_bkg,
                             skip_placeholder=False, nnz=None)

        yS, planes_sig = _event_from_root(
            args.root_file, args.tree, gi_sig, H, W,
            args.br_y, args.br_u, args.br_v, args.br_w,
            args.thr, args.signlog
        )
        yB, planes_bkg = _event_from_root(
            args.root_file, args.tree, gi_bkg, H, W,
            args.br_y, args.br_u, args.br_v, args.br_w,
            args.thr, args.signlog
        )

        if yS != 1 or yB != 0:
            print(f"[root] WARNING: label sanity check gave yS={yS} yB={yB}", file=sys.stderr)

        # Print quick diagnostics
        def _stats(ps):
            v = np.concatenate([p.reshape(-1) for p in ps])
            nz = v[v != 0]
            if nz.size == 0:
                return "all zero after transform"
            return f"nonzero={nz.size} min/mean/max={float(nz.min()):+.3e}/{float(nz.mean()):+.3e}/{float(nz.max()):+.3e}"

        print(f"[root] signal gi={gi_sig} thr={args.thr} signlog={args.signlog} :: {_stats(planes_sig)}")
        print(f"[root] bkg    gi={gi_bkg} thr={args.thr} signlog={args.signlog} :: {_stats(planes_bkg)}")

        title_sig = f"signal (label=1) gi={gi_sig}"
        title_bkg = f"bkg (label=0) gi={gi_bkg}"

    # Plot
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vmin, vmax = _robust_vmin_vmax(planes_sig + planes_bkg)
    names = ["U", "V", "W"]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)

    for j in range(3):
        ax = axes[0, j]
        im = ax.imshow(planes_sig[j], origin="upper", vmin=vmin, vmax=vmax)
        ax.set_title(f"{title_sig}  plane={names[j]}")
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(3):
        ax = axes[1, j]
        im = ax.imshow(planes_bkg[j], origin="upper", vmin=vmin, vmax=vmax)
        ax.set_title(f"{title_bkg}  plane={names[j]}")
        ax.set_xticks([])
        ax.set_yticks([])

    # One colorbar for the whole figure
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)

    fig.suptitle(f"source={args.source}  HxW={H}x{W}  vmin/vmax={vmin:+.2e}/{vmax:+.2e}")
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
