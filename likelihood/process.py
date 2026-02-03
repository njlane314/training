from __future__ import annotations

import gc
import os
import time
import warnings
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
import uproot

from . import config as cfg

_DUMMY_COORDS = np.array([[0, 0, 0]], dtype=np.int32)
_DUMMY_FEATS = np.array([[0.0, 0.0]], dtype=np.float16)

def _atomic_torch_save(obj, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(obj, tmp)
    tmp.replace(path)

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


def event_to_sparse(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    *,
    width: int,
    thresh: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    iu = np.flatnonzero(u > thresh)
    iv = np.flatnonzero(v > thresh)
    iw = np.flatnonzero(w > thresh)

    total = iu.size + iv.size + iw.size
    if total == 0:
        return _DUMMY_COORDS, _DUMMY_FEATS, 0

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

    return coords, feats, int(total)


def pack_events(coords_list: Iterable[np.ndarray], feats_list: Iterable[np.ndarray]):
    coords_list = list(coords_list)
    feats_list = list(feats_list)
    n_events = len(coords_list)
    lengths = np.fromiter((c.shape[0] for c in coords_list), dtype=np.int64, count=n_events)
    starts = np.empty(len(lengths) + 1, dtype=np.int64)
    starts[0] = 0
    np.cumsum(lengths, out=starts[1:])

    total = int(starts[-1])
    coords = np.empty((total, 3), dtype=np.int32)
    feats = np.empty((total, 2), dtype=np.float16)
    off = 0
    for c, f in zip(coords_list, feats_list):
        n = int(c.shape[0])
        coords[off : off + n] = c
        feats[off : off + n] = f
        off += n

    return (
        torch.from_numpy(coords),
        torch.from_numpy(feats),
        torch.from_numpy(starts),
    )

def _infer_hw_from_any(raw: np.ndarray, *, fallback_h: int, fallback_w: int) -> Tuple[int, int]:
    try:
        arr = np.asarray(raw)
    except Exception:
        return int(fallback_h), int(fallback_w)

    if arr.dtype != object:
        if arr.ndim >= 3:
            return int(arr.shape[-2]), int(arr.shape[-1])
        if arr.ndim == 2:
            if arr.shape[0] == fallback_h and arr.shape[1] == fallback_w:
                return int(fallback_h), int(fallback_w)
            hw = int(arr.shape[1])
            side = int(np.sqrt(hw))
            if side * side == hw:
                return side, side
            return int(fallback_h), int(fallback_w)
        if arr.ndim == 1:
            hw = int(arr.size)
            if hw == fallback_h * fallback_w:
                return int(fallback_h), int(fallback_w)
            side = int(np.sqrt(hw))
            if side * side == hw:
                return side, side
            return int(fallback_h), int(fallback_w)

    if arr.dtype == object and arr.ndim >= 1 and arr.shape[0] > 0:
        for item in arr:
            if item is None:
                continue
            try:
                x = np.asarray(item)
            except Exception:
                continue
            if x.ndim >= 2:
                return int(x.shape[-2]), int(x.shape[-1])
            hw = int(x.size)
            if hw == fallback_h * fallback_w:
                return int(fallback_h), int(fallback_w)
            side = int(np.sqrt(hw))
            if side * side == hw:
                return side, side
            break

    return int(fallback_h), int(fallback_w)

def _flatten_plane_batch(
    raw: np.ndarray,
    *,
    h: int,
    w: int,
    name: str,
    strict: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    hw = int(h) * int(w)

    arr = np.asarray(raw)

    if arr.dtype != object:
        if arr.ndim == 3 and arr.shape[-2:] == (h, w):
            flat = arr.reshape(arr.shape[0], hw).astype(np.float32, copy=False)
            return flat, np.zeros(flat.shape[0], dtype=bool)
        if arr.ndim == 2:
            if arr.shape[1] == hw:
                flat = arr.astype(np.float32, copy=False)
                return flat, np.zeros(flat.shape[0], dtype=bool)
            if arr.shape == (h, w):
                flat = arr.reshape(1, hw).astype(np.float32, copy=False)
                return flat, np.zeros(1, dtype=bool)
        if arr.ndim == 1 and arr.size == hw:
            flat = arr.reshape(1, hw).astype(np.float32, copy=False)
            return flat, np.zeros(1, dtype=bool)

    if arr.dtype == object:
        try:
            stacked = np.stack(arr)
        except Exception:
            stacked = None
        if stacked is not None and stacked.dtype != object:
            return _flatten_plane_batch(stacked, h=h, w=w, name=name, strict=strict)

    if arr.ndim == 0:
        msg = f"{name} has unexpected scalar shape {arr.shape}"
        if strict:
            raise ValueError(msg)
        warnings.warn(msg)
        return np.zeros((0, hw), dtype=np.float32), np.zeros((0,), dtype=bool)

    n = int(arr.shape[0])
    out = np.zeros((n, hw), dtype=np.float32)
    bad = np.zeros(n, dtype=bool)

    for i in range(n):
        item = arr[i]
        if item is None:
            bad[i] = True
            continue
        try:
            x = np.asarray(item)
        except Exception:
            bad[i] = True
            continue
        if x.size != hw:
            msg = f"{name}[{i}] has size {x.size} != {hw} (H*W)"
            if strict:
                raise ValueError(msg)
            bad[i] = True
            continue
        out[i] = x.reshape(-1).astype(np.float32, copy=False)

    return out, bad


def write_shards_from_root():
    out_dir = Path(cfg.SHARDS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.glob("shard_*.pt"):
        p.unlink(missing_ok=True)
    for p in out_dir.glob(".shard_*.pt.tmp.*"):
        p.unlink(missing_ok=True)
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

    with uproot.open(cfg.ROOT_FILE) as f:
        t = f[cfg.TREE]

        n_events = int(t.num_entries)
        labels = np.empty(n_events, dtype=np.uint8)
        weights = np.empty(n_events, dtype=np.float32)

        nnz = np.zeros(n_events, dtype=np.int32)
        bad_events: list[int] = []
        bad_logged = 0

        shard_id = 0
        shard_start = 0
        coords_acc, feats_acc = [], []
        n_acc = 0

        progress_every = max(1, n_events // 200)
        render_progress(0, n_events)

        h = int(cfg.H)
        w = int(cfg.W)
        thresh = float(cfg.THRESH)

        for start in range(0, n_events, cfg.CHUNK_EVENTS):
            stop = min(start + cfg.CHUNK_EVENTS, n_events)
            a = t.arrays(
                [cfg.BR_U, cfg.BR_V, cfg.BR_W, cfg.BR_Y, cfg.BR_WGT],
                entry_start=start,
                entry_stop=stop,
                library="np",
            )

            uu_raw, vv_raw, ww_raw = a[cfg.BR_U], a[cfg.BR_V], a[cfg.BR_W]
            if start == 0:
                hu, wu = _infer_hw_from_any(uu_raw, fallback_h=h, fallback_w=w)
                hv, wv = _infer_hw_from_any(vv_raw, fallback_h=h, fallback_w=w)
                hw_, ww_ = _infer_hw_from_any(ww_raw, fallback_h=h, fallback_w=w)
                if (hu, wu) == (hv, wv) == (hw_, ww_) and (hu, wu) != (h, w):
                    warnings.warn(
                        f"cfg.H/cfg.W={h}x{w} do not match data={hu}x{wu}; "
                        f"using inferred H/W from file"
                    )
                    h, w = int(hu), int(wu)

            y = np.asarray(a[cfg.BR_Y]).reshape(-1)
            if y.size != (stop - start):
                msg = f"{cfg.BR_Y} batch has size {y.size} != {stop-start}; zero-filling"
                if cfg.STRICT_SHAPES:
                    raise ValueError(msg)
                if bad_logged < cfg.MAX_BAD_EVENT_LOG:
                    warnings.warn(msg)
                y = np.zeros(stop - start, dtype=np.uint8)
            labels[start:stop] = y.astype(np.uint8, copy=False)

            wgt = np.asarray(a[cfg.BR_WGT]).reshape(-1)
            if wgt.size != (stop - start):
                msg = f"{cfg.BR_WGT} batch has size {wgt.size} != {stop-start}; one-filling"
                if cfg.STRICT_SHAPES:
                    raise ValueError(msg)
                if bad_logged < cfg.MAX_BAD_EVENT_LOG:
                    warnings.warn(msg)
                wgt = np.ones(stop - start, dtype=np.float32)
            weights[start:stop] = wgt.astype(np.float32, copy=False)

            uu, bad_u = _flatten_plane_batch(
                uu_raw, h=h, w=w, name=cfg.BR_U, strict=cfg.STRICT_SHAPES
            )
            vv, bad_v = _flatten_plane_batch(
                vv_raw, h=h, w=w, name=cfg.BR_V, strict=cfg.STRICT_SHAPES
            )
            ww, bad_w = _flatten_plane_batch(
                ww_raw, h=h, w=w, name=cfg.BR_W, strict=cfg.STRICT_SHAPES
            )
            bad_batch = bad_u | bad_v | bad_w

            for j in range(stop - start):
                gi = start + j
                try:
                    c, fe, nnz_true = event_to_sparse(
                        uu[j], vv[j], ww[j], width=w, thresh=thresh
                    )
                except Exception as e:
                    if cfg.STRICT_SHAPES:
                        raise
                    c, fe, nnz_true = _DUMMY_COORDS, _DUMMY_FEATS, 0
                    bad_batch[j] = True
                    if bad_logged < cfg.MAX_BAD_EVENT_LOG:
                        warnings.warn(f"event {gi}: event_to_sparse failed ({type(e).__name__}: {e}); zero-filling")
                        bad_logged += 1

                nnz[gi] = int(nnz_true)
                if bad_batch[j]:
                    bad_events.append(int(gi))

                coords_acc.append(c)
                feats_acc.append(fe)
                n_acc += 1

                if n_acc == cfg.SHARD_EVENTS:
                    coords_t, feats_t, starts_t = pack_events(coords_acc, feats_acc)
                    _atomic_torch_save(
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
                    gc.collect()

                if (gi + 1) % progress_every == 0 or (gi + 1) == n_events:
                    render_progress(gi + 1, n_events)

        if n_acc:
            coords_t, feats_t, starts_t = pack_events(coords_acc, feats_acc)
            _atomic_torch_save(
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

        _atomic_torch_save(
            {
                "H": int(h),
                "W": int(w),
                "shard_events": int(cfg.SHARD_EVENTS),
                "n_events": int(n_events),
                "labels": torch.from_numpy(labels),
                "weights": torch.from_numpy(weights),
                "nnz": torch.from_numpy(nnz),
                "bad_events": torch.tensor(bad_events, dtype=torch.int64),
                "branches": {"y": cfg.BR_Y, "u": cfg.BR_U, "v": cfg.BR_V, "w": cfg.BR_W, "wgt": cfg.BR_WGT},
            },
            idx_path,
        )

    print()
    print(f"wrote {shard_id} shards to {out_dir} (events={n_events})")
    if bad_events:
        print(f"warning: {len(bad_events)} events had missing/malformed data and were zero-filled (see index.pt: bad_events)")
