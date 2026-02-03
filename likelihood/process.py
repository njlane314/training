# make_shards.py
import glob
import os
import time
from pathlib import Path

import numpy as np
import torch
import uproot

from . import config as cfg

_DUMMY_COORDS = np.array([[0, 0, 0]], dtype=np.int32)
_DUMMY_FEATS = np.array([[0.0, 0.0]], dtype=np.float16)

def plane_to_sparse(
    flat: np.ndarray,
    plane: int,
    *,
    h: int = cfg.H,
    w: int = cfg.W,
    thresh: float = cfg.THRESH,
):
    flat = np.asarray(flat, dtype=np.float32).ravel()
    if flat.size != h * w:
        raise ValueError(f"plane size {flat.size} != {h}*{w}")

    idx = np.flatnonzero(flat > thresh)
    if idx.size == 0:
        return None, None

    val = flat[idx]
    y, x = np.divmod(idx, w)

    coords = np.empty((idx.size, 3), dtype=np.int32)
    coords[:, 0] = plane
    coords[:, 1] = y.astype(np.int32, copy=False)
    coords[:, 2] = x.astype(np.int32, copy=False)

    # Features: occupancy, log-charge
    feats = np.empty((idx.size, 2), dtype=np.float32)
    feats[:, 0] = 1.0
    np.maximum(val, 0.0, out=val)
    np.log1p(val, out=feats[:, 1])

    return coords, feats


def event_to_sparse(u, v, w, *, width: int = cfg.W):
    iu = np.flatnonzero(u > cfg.THRESH)
    iv = np.flatnonzero(v > cfg.THRESH)
    iw = np.flatnonzero(w > cfg.THRESH)

    total = iu.size + iv.size + iw.size
    if total == 0:
        return _DUMMY_COORDS, _DUMMY_FEATS

    coords = np.empty((total, 3), dtype=np.int32)
    feats = np.empty((total, 2), dtype=np.float16)

    off = 0
    for plane, (arr, idx) in enumerate(((u, iu), (v, iv), (w, iw))):
        n = idx.size
        if n == 0:
            continue

        sl = slice(off, off + n)
        coords[sl, 0] = plane
        np.floor_divide(idx, width, out=coords[sl, 1], casting="unsafe")
        np.remainder(idx, width, out=coords[sl, 2], casting="unsafe")

        feats[sl, 0] = 1.0
        vals = arr[idx]
        np.maximum(vals, 0.0, out=vals)
        np.log1p(vals, out=vals)
        feats[sl, 1] = vals

        off += n

    return coords, feats


def pack_events(coords_list, feats_list):
    lengths = np.fromiter((c.shape[0] for c in coords_list), dtype=np.int64, count=len(coords_list))
    starts = np.empty(len(lengths) + 1, dtype=np.int64)
    starts[0] = 0
    np.cumsum(lengths, out=starts[1:])

    coords = np.concatenate(coords_list, axis=0)
    feats = np.concatenate(feats_list, axis=0)

    return (
        torch.from_numpy(coords),
        torch.from_numpy(feats),
        torch.from_numpy(starts),
    )


def write_shards_from_root():
    out_dir = Path(cfg.SHARDS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in glob.glob(str(out_dir / "shard_*.pt")):
        os.remove(p)
    idx_path = out_dir / "index.pt"
    if idx_path.exists():
        idx_path.unlink()

    start_time = time.monotonic()

    def format_eta(seconds: float) -> str:
        if not np.isfinite(seconds) or seconds < 0:
            return "--:--:--"
        minutes, sec = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"

    def render_progress(current: int, total: int, width: int = 40) -> None:
        if total <= 0:
            return
        ratio = min(max(current / total, 0.0), 1.0)
        filled = int(ratio * width)
        bar = "=" * filled + "-" * (width - filled)
        if current > 0:
            elapsed = time.monotonic() - start_time
            remaining = (total - current) * (elapsed / current)
            eta = format_eta(remaining)
        else:
            eta = "--:--:--"
        print(f"\rProcessing events: |{bar}| {current}/{total} ETA {eta}", end="", flush=True)

    def infer_hw(arr: np.ndarray, *, name: str):
        arr = np.asarray(arr)
        if arr.ndim >= 3:
            h, w = arr.shape[-2], arr.shape[-1]
            return arr.reshape(arr.shape[0], h * w), int(h), int(w)
        if arr.ndim == 2:
            flat = arr
        elif arr.ndim == 1:
            flat = arr.reshape(1, -1)
        else:
            raise ValueError(f"{name} has unexpected shape {arr.shape}")

        size = int(flat.shape[1])
        if size == cfg.H * cfg.W:
            return flat, int(cfg.H), int(cfg.W)
        side = int(np.sqrt(size))
        if side * side != size:
            raise ValueError(f"{name} plane size {size} is not square and doesn't match H*W={cfg.H * cfg.W}")
        return flat, side, side

    with uproot.open(cfg.ROOT_FILE) as f:
        t = f[cfg.TREE]

        n_events = int(t.num_entries)
        labels = np.empty(n_events, dtype=np.uint8)
        weights = np.empty(n_events, dtype=np.float32)

        nnz = np.zeros(n_events, dtype=np.int32)

        shard_id = 0
        shard_start = 0
        coords_acc, feats_acc = [], []
        n_acc = 0

        progress_every = max(1, n_events // 200)
        render_progress(0, n_events)

        for start in range(0, n_events, cfg.CHUNK_EVENTS):
            stop = min(start + cfg.CHUNK_EVENTS, n_events)
            a = t.arrays(
                [cfg.BR_U, cfg.BR_V, cfg.BR_W, cfg.BR_Y, cfg.BR_WGT],
                entry_start=start,
                entry_stop=stop,
                library="np",
            )

            uu_raw, vv_raw, ww_raw = a[cfg.BR_U], a[cfg.BR_V], a[cfg.BR_W]
            labels[start:stop] = a[cfg.BR_Y].astype(np.uint8, copy=False).reshape(-1)
            weights[start:stop] = a[cfg.BR_WGT].astype(np.float32, copy=False).reshape(-1)

            uu, h, w = infer_hw(uu_raw, name=cfg.BR_U)
            vv, h_v, w_v = infer_hw(vv_raw, name=cfg.BR_V)
            ww, h_w, w_w = infer_hw(ww_raw, name=cfg.BR_W)
            if (h, w) != (h_v, w_v) or (h, w) != (h_w, w_w):
                raise ValueError(
                    f"plane sizes differ: u={h}x{w}, v={h_v}x{w_v}, w={h_w}x{w_w}"
                )

            uu = uu.astype(np.float32, copy=False)
            vv = vv.astype(np.float32, copy=False)
            ww = ww.astype(np.float32, copy=False)

            for j in range(stop - start):
                gi = start + j
                c, fe = event_to_sparse(uu[j], vv[j], ww[j], width=w)
                nnz[gi] = int(c.shape[0])

                coords_acc.append(c)
                feats_acc.append(fe)
                n_acc += 1

                if n_acc == cfg.SHARD_EVENTS:
                    coords_t, feats_t, starts_t = pack_events(coords_acc, feats_acc)
                    torch.save(
                        {
                            "start_event": int(shard_start),
                            "n_events": int(n_acc),
                            "coords": coords_t,
                            "feats": feats_t,
                            "starts": starts_t,
                        },
                        out_dir / f"shard_{shard_id:05d}.pt",
                    )
                    shard_id += 1
                    shard_start = gi + 1
                    coords_acc.clear()
                    feats_acc.clear()
                    n_acc = 0

                if (gi + 1) % progress_every == 0 or (gi + 1) == n_events:
                    render_progress(gi + 1, n_events)

        if n_acc:
            coords_t, feats_t, starts_t = pack_events(coords_acc, feats_acc)
            torch.save(
                {
                    "start_event": int(shard_start),
                    "n_events": int(n_acc),
                    "coords": coords_t,
                    "feats": feats_t,
                    "starts": starts_t,
                },
                out_dir / f"shard_{shard_id:05d}.pt",
            )
            shard_id += 1

        torch.save(
            {
                "H": int(h),
                "W": int(w),
                "shard_events": int(cfg.SHARD_EVENTS),
                "n_events": int(n_events),
                "labels": torch.from_numpy(labels),
                "weights": torch.from_numpy(weights),
                "nnz": torch.from_numpy(nnz),
                "branches": {"y": cfg.BR_Y, "u": cfg.BR_U, "v": cfg.BR_V, "w": cfg.BR_W, "wgt": cfg.BR_WGT},
            },
            idx_path,
        )

    print()
    print(f"wrote {shard_id} shards to {out_dir} (events={n_events})")
