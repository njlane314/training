# make_shards.py
import glob
import os
from pathlib import Path

import numpy as np
import torch
import uproot

from . import config as cfg

def plane_to_sparse(flat: np.ndarray, plane: int):
    flat = np.asarray(flat, dtype=np.float32).reshape(-1)
    if flat.size != cfg.H * cfg.W:
        raise ValueError(f"plane size {flat.size} != {cfg.H}*{cfg.W}")

    idx = np.flatnonzero(flat > cfg.THRESH)
    if idx.size == 0:
        return None, None

    val = flat[idx]
    y, x = np.divmod(idx.astype(np.int64), cfg.W)

    coords = np.empty((idx.size, 3), dtype=np.int32)
    coords[:, 0] = int(plane)
    coords[:, 1] = y.astype(np.int32, copy=False)
    coords[:, 2] = x.astype(np.int32, copy=False)

    # Features: occupancy, log-charge
    feats = np.empty((idx.size, 2), dtype=np.float32)
    feats[:, 0] = 1.0
    feats[:, 1] = np.log1p(np.maximum(val, 0.0))

    return coords, feats


def event_to_sparse(u, v, w):
    coords_list, feats_list = [], []
    for plane, arr in enumerate((u, v, w)):
        c, f = plane_to_sparse(arr, plane)
        if c is not None:
            coords_list.append(c)
            feats_list.append(f)

    if not coords_list:
        # single dummy site (keeps downstream code simple)
        coords = np.array([[0, 0, 0]], dtype=np.int32)
        feats = np.array([[0.0, 0.0]], dtype=np.float32)
        return coords, feats

    return np.concatenate(coords_list, axis=0), np.concatenate(feats_list, axis=0)


def pack_events(coords_list, feats_list):
    n = len(coords_list)
    starts = np.zeros(n + 1, dtype=np.int64)
    for i, c in enumerate(coords_list):
        starts[i + 1] = starts[i] + int(c.shape[0])

    coords = np.concatenate(coords_list, axis=0).astype(np.int32, copy=False)
    feats = np.concatenate(feats_list, axis=0).astype(np.float16, copy=False)

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

    with uproot.open(cfg.ROOT_FILE) as f:
        t = f[cfg.TREE]

        labels = t[cfg.BR_Y].array(library="np").astype(np.uint8).reshape(-1)
        weights = t[cfg.BR_WGT].array(library="np").astype(np.float32).reshape(-1)
        n_events = int(labels.shape[0])

        nnz = np.zeros(n_events, dtype=np.int32)

        shard_id = 0
        shard_start = 0
        coords_acc, feats_acc = [], []
        n_acc = 0

        for start in range(0, n_events, cfg.CHUNK_EVENTS):
            stop = min(start + cfg.CHUNK_EVENTS, n_events)
            a = t.arrays([cfg.BR_U, cfg.BR_V, cfg.BR_W], entry_start=start, entry_stop=stop, library="np")

            uu, vv, ww = a[cfg.BR_U], a[cfg.BR_V], a[cfg.BR_W]

            for j in range(stop - start):
                gi = start + j
                c, fe = event_to_sparse(uu[j], vv[j], ww[j])
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
                "H": int(cfg.H),
                "W": int(cfg.W),
                "shard_events": int(cfg.SHARD_EVENTS),
                "n_events": int(n_events),
                "labels": torch.from_numpy(labels),
                "weights": torch.from_numpy(weights),
                "nnz": torch.from_numpy(nnz),
                "branches": {"y": cfg.BR_Y, "u": cfg.BR_U, "v": cfg.BR_V, "w": cfg.BR_W, "wgt": cfg.BR_WGT},
            },
            idx_path,
        )

    print(f"wrote {shard_id} shards to {out_dir} (events={n_events})")
