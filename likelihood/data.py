# make_shards.py
import glob
import os
from pathlib import Path

import numpy as np
import torch
import uproot

# -------------------------
# Edit these few constants
# -------------------------
ROOT_FILE = "your_file.root"
TREE = "your_tree"

BR_U = "detector_image_u"
BR_V = "detector_image_v"
BR_W = "detector_image_w"
BR_Y = "is_signal"     # 0/1
BR_WGT = "w_nominal"   # float

H, W = 512, 512
THRESH = 0.0           # pixels <= THRESH are dropped
CHUNK_EVENTS = 512
SHARD_EVENTS = 2048
OUT_DIR = Path("shards")


def plane_to_sparse(flat: np.ndarray, plane: int):
    flat = np.asarray(flat, dtype=np.float32).reshape(-1)
    if flat.size != H * W:
        raise ValueError(f"plane size {flat.size} != {H}*{W}")

    idx = np.flatnonzero(flat > THRESH)
    if idx.size == 0:
        return None, None

    val = flat[idx]
    y, x = np.divmod(idx.astype(np.int64), W)

    coords = np.empty((idx.size, 3), dtype=np.int32)
    coords[:, 0] = int(plane)
    coords[:, 1] = y.astype(np.int32, copy=False)
    coords[:, 2] = x.astype(np.int32, copy=False)

    # Features: occupancy, log-charge, plane-id
    feats = np.empty((idx.size, 3), dtype=np.float32)
    feats[:, 0] = 1.0
    feats[:, 1] = np.log1p(np.maximum(val, 0.0))
    feats[:, 2] = float(plane - 1)  # {-1,0,+1} for {U,V,W}

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
        feats = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in glob.glob(str(OUT_DIR / "shard_*.pt")):
        os.remove(p)
    idx_path = OUT_DIR / "index.pt"
    if idx_path.exists():
        idx_path.unlink()

    with uproot.open(ROOT_FILE) as f:
        t = f[TREE]

        labels = t[BR_Y].array(library="np").astype(np.uint8).reshape(-1)
        weights = t[BR_WGT].array(library="np").astype(np.float32).reshape(-1)
        n_events = int(labels.shape[0])

        nnz = np.zeros(n_events, dtype=np.int32)

        shard_id = 0
        shard_start = 0
        coords_acc, feats_acc, y_acc = [], [], []

        for start in range(0, n_events, CHUNK_EVENTS):
            stop = min(start + CHUNK_EVENTS, n_events)
            a = t.arrays([BR_U, BR_V, BR_W], entry_start=start, entry_stop=stop, library="np")

            uu, vv, ww = a[BR_U], a[BR_V], a[BR_W]

            for j in range(stop - start):
                gi = start + j
                c, fe = event_to_sparse(uu[j], vv[j], ww[j])
                nnz[gi] = int(c.shape[0])

                coords_acc.append(c)
                feats_acc.append(fe)
                y_acc.append(int(labels[gi]))

                if len(y_acc) == SHARD_EVENTS:
                    coords_t, feats_t, starts_t = pack_events(coords_acc, feats_acc)
                    torch.save(
                        {
                            "start_event": int(shard_start),
                            "n_events": int(len(y_acc)),
                            "coords": coords_t,
                            "feats": feats_t,
                            "starts": starts_t,
                            "labels": torch.tensor(y_acc, dtype=torch.uint8),
                        },
                        OUT_DIR / f"shard_{shard_id:05d}.pt",
                    )
                    shard_id += 1
                    shard_start = gi + 1
                    coords_acc.clear()
                    feats_acc.clear()
                    y_acc.clear()

        if y_acc:
            coords_t, feats_t, starts_t = pack_events(coords_acc, feats_acc)
            torch.save(
                {
                    "start_event": int(shard_start),
                    "n_events": int(len(y_acc)),
                    "coords": coords_t,
                    "feats": feats_t,
                    "starts": starts_t,
                    "labels": torch.tensor(y_acc, dtype=torch.uint8),
                },
                OUT_DIR / f"shard_{shard_id:05d}.pt",
            )
            shard_id += 1

        torch.save(
            {
                "H": int(H),
                "W": int(W),
                "shard_events": int(SHARD_EVENTS),
                "n_events": int(n_events),
                "labels": torch.from_numpy(labels),
                "weights": torch.from_numpy(weights),
                "nnz": torch.from_numpy(nnz),
                "branches": {"y": BR_Y, "u": BR_U, "v": BR_V, "w": BR_W, "wgt": BR_WGT},
            },
            idx_path,
        )

    print(f"wrote {shard_id} shards to {OUT_DIR} (events={n_events})")


if __name__ == "__main__":
    main()
