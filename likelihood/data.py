import glob
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
import uproot

from . import config as cfg

BR_Y = os.environ.get("BR_Y", "is_signal")
BR_U = os.environ.get("BR_U", "detector_image_u")
BR_V = os.environ.get("BR_V", "detector_image_v")
BR_W = os.environ.get("BR_W", "detector_image_w")
BR_WGT = os.environ.get("BR_WGT", "w_nominal")


def plane_to_sparse(flat, view, H, W, thr, signlog):
    """
    @brief Transform a flattened plane into sparse coordinates with per-hit features.
    """
    flat = np.asarray(flat, dtype=np.float32).reshape(-1)
    if flat.size != H * W:
        raise ValueError(f"plane size {flat.size} != {H}*{W}")

    if signlog:
        idx = np.flatnonzero(flat if thr <= 0.0 else (np.abs(flat) > thr))
        val = flat[idx]
        adc = np.sign(val) * np.log1p(np.abs(val))
    else:
        idx = np.flatnonzero(flat > thr)
        val = flat[idx]
        adc = np.log1p(np.maximum(val, 0.0))

    if idx.size == 0:
        return None, None

    y, x = np.divmod(idx.astype(np.int64, copy=False), W)
    y_norm = (y.astype(np.float32, copy=False) - (H / 2.0)) / (H / 2.0)
    x_norm = (x.astype(np.float32, copy=False) - (W / 2.0)) / (W / 2.0)
    v_norm = np.full_like(y_norm, (float(view) - 1.0), dtype=np.float32)

    coords = np.empty((idx.size, 3), dtype=np.int32)
    coords[:, 0] = int(view)
    coords[:, 1] = y.astype(np.int32, copy=False)
    coords[:, 2] = x.astype(np.int32, copy=False)

    feats = np.empty((idx.size, 4), dtype=np.float32)
    feats[:, 0] = adc.astype(np.float32, copy=False)
    feats[:, 1] = y_norm
    feats[:, 2] = x_norm
    feats[:, 3] = v_norm
    return coords, feats


def event_to_sparse(u, v, w, H, W, thr, signlog):
    """
    @brief Merge U/V/W planes into a sparse event representation.
    """
    coords = []
    feats = []
    for view, flat in enumerate((u, v, w)):
        c, f = plane_to_sparse(flat, view, H, W, thr, signlog)
        if c is not None:
            coords.append(c)
            feats.append(f)
    if not coords:
        return np.array([[0, 0, 0]], dtype=np.int32), np.zeros((1, 4), dtype=np.float32)
    return np.concatenate(coords, axis=0), np.concatenate(feats, axis=0)


def pack_events(coords_list, feats_list, feat_dtype=np.float16):
    """
    @brief Pack per-event sparse tensors into a batched shard layout for serialisation.
    """
    n = len(coords_list)
    starts = np.empty(n + 1, dtype=np.int64)
    starts[0] = 0
    for i, c in enumerate(coords_list):
        starts[i + 1] = starts[i] + int(c.shape[0])
    coords = np.concatenate(coords_list, axis=0).astype(np.int32, copy=False)
    feats = np.concatenate(feats_list, axis=0).astype(feat_dtype, copy=False)
    return torch.from_numpy(coords), torch.from_numpy(feats), torch.from_numpy(starts)


def write_shards_from_root():
    """
    @brief Write sparse shards from the configured ROOT file with progress reporting.
    """
    out_dir = cfg.SHARDS_OUT
    os.makedirs(out_dir, exist_ok=True)
    for p in glob.glob(os.path.join(out_dir, "shard_*.pt")):
        os.remove(p)
    idx_path = os.path.join(out_dir, "index.pt")
    if os.path.exists(idx_path):
        os.remove(idx_path)

    def render_progress(processed, total, width=32):
        """
        @brief Render a simple progress bar to stdout.
        """
        if total <= 0:
            return
        ratio = min(max(processed / total, 0.0), 1.0)
        filled = int(width * ratio)
        bar = "=" * filled + "-" * (width - filled)
        msg = f"\rProcessing events [{bar}] {processed}/{total} ({ratio:.1%})"
        print(msg, end="", file=sys.stdout, flush=True)

    with uproot.open(cfg.ROOT_FILE) as f:
        t = f[cfg.TREE]
        labels = t[BR_Y].array(library="np").astype(np.uint8).reshape(-1)
        weights = t[BR_WGT].array(library="np").astype(np.float32).reshape(-1)
        n_events = int(labels.shape[0])

        nnz = np.zeros(n_events, dtype=np.int32)

        shard_id = 0
        shard_start = 0
        coords_list = []
        feats_list = []
        y_local = []

        for start in range(0, n_events, cfg.CHUNK_EVENTS):
            stop = min(start + cfg.CHUNK_EVENTS, n_events)
            a = t.arrays([BR_U, BR_V, BR_W], entry_start=start, entry_stop=stop, library="np")
            uu = a[BR_U]
            vv = a[BR_V]
            ww = a[BR_W]

            for j in range(stop - start):
                gi = start + j
                c, fe = event_to_sparse(uu[j], vv[j], ww[j], cfg.H, cfg.W, cfg.THRESH, cfg.ADC_SIGNLOG)
                nnz[gi] = int(c.shape[0])
                coords_list.append(c)
                feats_list.append(fe)
                y_local.append(int(labels[gi]))

                if len(y_local) == cfg.SHARD_EVENTS:
                    coords_t, feats_t, starts_t = pack_events(coords_list, feats_list, feat_dtype=np.float16)
                    shard_path = os.path.join(out_dir, f"shard_{shard_id:05d}.pt")
                    torch.save(
                        {
                            "start_event": int(shard_start),
                            "n_events": int(len(y_local)),
                            "coords": coords_t,
                            "feats": feats_t,
                            "starts": starts_t,
                            "labels": torch.tensor(y_local, dtype=torch.uint8),
                        },
                        shard_path,
                    )
                    shard_id += 1
                    shard_start = gi + 1
                    coords_list.clear()
                    feats_list.clear()
                    y_local.clear()
            render_progress(stop, n_events)

        if y_local:
            coords_t, feats_t, starts_t = pack_events(coords_list, feats_list, feat_dtype=np.float16)
            shard_path = os.path.join(out_dir, f"shard_{shard_id:05d}.pt")
            torch.save(
                {
                    "start_event": int(shard_start),
                    "n_events": int(len(y_local)),
                    "coords": coords_t,
                    "feats": feats_t,
                    "starts": starts_t,
                    "labels": torch.tensor(y_local, dtype=torch.uint8),
                },
                shard_path,
            )
            shard_id += 1

    if n_events:
        print(file=sys.stdout, flush=True)

    torch.save(
        {
            "H": int(cfg.H),
            "W": int(cfg.W),
            "shard_events": int(cfg.SHARD_EVENTS),
            "n_events": int(n_events),
            "labels": torch.from_numpy(labels),
            "weights": torch.from_numpy(weights),
            "nnz": torch.from_numpy(nnz),
            "branches": {"y": BR_Y, "u": BR_U, "v": BR_V, "w": BR_W, "wgt": BR_WGT},
        },
        idx_path,
    )

    print(f"wrote {shard_id} shards to {out_dir} (events={n_events})", flush=True)


class ShardDataset(torch.utils.data.Dataset):
    """
    @brief Dataset that loads sparse events from shard files with a small in-memory cache.
    """

    def __init__(self, shards_dir, event_indices, cache_size=2):
        """
        @brief Initialise dataset metadata, indices, and the shard cache.
        """
        meta = torch.load(os.path.join(shards_dir, "index.pt"), map_location="cpu")
        self.shards_dir = shards_dir
        self.shard_events = int(meta["shard_events"])
        self.labels_all = np.asarray(meta["labels"], dtype=np.uint8)

        self.event_indices = np.asarray(event_indices, dtype=np.int64)
        self.labels = self.labels_all[self.event_indices].astype(np.uint8, copy=False)

        self.shard_ids = (self.event_indices // self.shard_events).astype(np.int64, copy=False)
        self.local_ids = (self.event_indices - self.shard_ids * self.shard_events).astype(np.int64, copy=False)

        self.cache_size = int(cache_size)
        self._cache = OrderedDict()

    def __len__(self):
        """
        @brief Return the number of indexed events.
        """
        return int(self.event_indices.shape[0])

    def _load_shard(self, sid):
        """
        @brief Load a shard file into the cache, evicting old entries if needed.
        """
        sid = int(sid)
        if sid in self._cache:
            self._cache.move_to_end(sid)
            return self._cache[sid]
        path = os.path.join(self.shards_dir, f"shard_{sid:05d}.pt")
        d = torch.load(path, map_location="cpu")
        self._cache[sid] = d
        self._cache.move_to_end(sid)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return d

    @staticmethod
    def _slice_one(d, local):
        """
        @brief Slice a single event from a shard dictionary.
        """
        local = int(local)
        s = int(d["starts"][local].item())
        e = int(d["starts"][local + 1].item())
        c = d["coords"][s:e]
        f = d["feats"][s:e].to(dtype=torch.float32)
        return c, f

    def __getitem__(self, i):
        """
        @brief Retrieve one event by dataset index.
        """
        gi = int(self.event_indices[i])
        sid = int(gi // self.shard_events)
        d = self._load_shard(sid)
        local = gi - int(d["start_event"])
        c, f = self._slice_one(d, local)
        y = float(self.labels[i])
        return c, f, y

    def __getitems__(self, idxs):
        """
        @brief Retrieve multiple events by dataset indices.
        """
        idxs = np.asarray(idxs, dtype=np.int64)
        gi = self.event_indices[idxs]
        sid = (gi // self.shard_events).astype(np.int64, copy=False)
        order = np.argsort(sid, kind="stable")

        out = [None] * int(idxs.shape[0])
        k = 0
        while k < order.size:
            sid_k = int(sid[order[k]])
            d = self._load_shard(sid_k)
            start_event = int(d["start_event"])
            j = k + 1
            while j < order.size and int(sid[order[j]]) == sid_k:
                j += 1
            for p in range(k, j):
                pos = int(order[p])
                gi_p = int(gi[pos])
                local = gi_p - start_event
                c, f = self._slice_one(d, local)
                y = float(self.labels[int(idxs[pos])])
                out[pos] = (c, f, y)
            k = j
        return out


class BalancedBatchSampler(torch.utils.data.Sampler):
    """
    @brief Sampler that yields balanced signal/background batches for stable training.
    """

    def __init__(self, labels, shard_ids, local_ids, batch_size, seed, steps=None):
        """
        @brief Initialise balanced batch sampling state.
        """
        if batch_size % 2:
            raise ValueError("batch_size must be even")
        self.bs = int(batch_size)
        self.h = self.bs // 2
        self.labels = np.asarray(labels, dtype=np.uint8)
        self.shard_ids = np.asarray(shard_ids, dtype=np.int64)
        self.local_ids = np.asarray(local_ids, dtype=np.int64)
        self.sig = np.where(self.labels == 1)[0]
        self.bkg = np.where(self.labels == 0)[0]
        if self.sig.size == 0 or self.bkg.size == 0:
            raise ValueError("need both classes")
        self.steps = int(steps) if steps is not None else int(np.ceil(max(self.sig.size, self.bkg.size) / self.h))
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, e):
        """
        @brief Set the sampler epoch for deterministic shuffling.
        """
        self.epoch = int(e)

    def __len__(self):
        """
        @brief Return the number of batches per epoch.
        """
        return self.steps

    def __iter__(self):
        """
        @brief Yield balanced batches of indices in a deterministic order per epoch.
        """
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.steps):
            s = rng.choice(self.sig, size=self.h, replace=True)
            b = rng.choice(self.bkg, size=self.h, replace=True)
            batch = np.concatenate([s, b]).astype(np.int64, copy=False)
            key = np.lexsort((self.local_ids[batch], self.shard_ids[batch]))
            yield batch[key].tolist()


def collate(batch):
    """
    @brief Collate sparse event tuples into MinkowskiEngine inputs and targets.
    """
    import MinkowskiEngine as ME

    coords, feats = ME.utils.sparse_collate([b[0] for b in batch], [b[1] for b in batch])
    y = torch.tensor([b[2] for b in batch], dtype=torch.float32).view(-1, 1)
    return coords, feats, y
