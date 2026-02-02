# lar_dataset.py
from collections import OrderedDict

import numpy as np
import torch

class ShardDataset(torch.utils.data.Dataset):
    """
    Loads sparse events from shard files created by make_shards.py.
    """

    def __init__(self, shards_dir: str, event_indices: np.ndarray, cache_size: int = 2):
        meta = torch.load(f"{shards_dir}/index.pt", map_location="cpu")

        self.shards_dir = shards_dir
        self.shard_events = int(meta["shard_events"])

        self.labels_all = np.asarray(meta["labels"], dtype=np.uint8)
        self.weights_all = np.asarray(meta["weights"], dtype=np.float32)

        self.event_indices = np.asarray(event_indices, dtype=np.int64)
        self.labels = self.labels_all[self.event_indices].astype(np.uint8, copy=False)
        self.weights = self.weights_all[self.event_indices].astype(np.float32, copy=False)

        # For IO-friendly sorting in the sampler
        self.shard_ids = (self.event_indices // self.shard_events).astype(np.int64, copy=False)
        self.local_ids = (self.event_indices - self.shard_ids * self.shard_events).astype(np.int64, copy=False)

        self.cache_size = int(cache_size)
        self._cache = OrderedDict()

    def __len__(self):
        return int(self.event_indices.shape[0])

    def _load_shard(self, sid: int):
        sid = int(sid)
        if sid in self._cache:
            self._cache.move_to_end(sid)
            return self._cache[sid]

        d = torch.load(f"{self.shards_dir}/shard_{sid:05d}.pt", map_location="cpu")
        self._cache[sid] = d
        self._cache.move_to_end(sid)

        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

        return d

    @staticmethod
    def _slice_one(d, local: int):
        local = int(local)
        s = int(d["starts"][local].item())
        e = int(d["starts"][local + 1].item())
        coords = d["coords"][s:e].to(dtype=torch.int32)
        feats = d["feats"][s:e].to(dtype=torch.float32)  # stored float16 on disk
        return coords, feats

    def __getitem__(self, i: int):
        gi = int(self.event_indices[i])
        sid = int(gi // self.shard_events)

        d = self._load_shard(sid)
        local = gi - int(d["start_event"])

        coords, feats = self._slice_one(d, local)
        y = float(self.labels[i])
        return coords, feats, y


class BalancedBatchSampler(torch.utils.data.Sampler):
    """
    Infinite i.i.d. sampling:
      - pick exactly half sig, half bkg per batch
      - within each class, sample proportional to nominal event weight
      - sort by (shard_id, local_id) to reduce shard thrash
    """

    def __init__(self, dataset: ShardDataset, batch_size: int, seed: int = 123):
        if batch_size % 2 != 0:
            raise ValueError("batch_size must be even")
        self.ds = dataset
        self.h = batch_size // 2
        self.rng = np.random.default_rng(int(seed))

        labels = np.asarray(self.ds.labels, dtype=np.uint8)
        w = np.asarray(self.ds.weights, dtype=np.float64)
        w = np.clip(w, 0.0, None)

        self.sig = np.flatnonzero(labels == 1)
        self.bkg = np.flatnonzero(labels == 0)
        if self.sig.size == 0 or self.bkg.size == 0:
            raise ValueError("need both signal and background in the training split")

        ws = w[self.sig]
        wb = w[self.bkg]
        if ws.sum() <= 0 or wb.sum() <= 0:
            raise ValueError("weights must sum to >0 within each class")

        self.ps = (ws / ws.sum()).astype(np.float64, copy=False)
        self.pb = (wb / wb.sum()).astype(np.float64, copy=False)

    def __iter__(self):
        while True:
            s = self.rng.choice(self.sig, size=self.h, replace=True, p=self.ps)
            b = self.rng.choice(self.bkg, size=self.h, replace=True, p=self.pb)
            batch = np.concatenate([s, b]).astype(np.int64, copy=False)

            # IO-friendly ordering
            key = np.lexsort((self.ds.local_ids[batch], self.ds.shard_ids[batch]))
            yield batch[key].tolist()


def collate_me(batch):
    coords_list, feats_list = [], []
    ys = []

    for bi, (c, f, y) in enumerate(batch):
        c = torch.as_tensor(c, dtype=torch.int32).contiguous()   # (N,3): (plane,y,x)
        f = torch.as_tensor(f, dtype=torch.float32).contiguous() # (N,3): (occ, logq, plane_id)
        bcol = torch.full((c.shape[0], 1), bi, dtype=torch.int32)
        coords_list.append(torch.cat([bcol, c], dim=1))
        feats_list.append(f)
        ys.append(y)

    coords = torch.cat(coords_list, dim=0).contiguous()  # (sumN,4)
    feats = torch.cat(feats_list, dim=0).contiguous()    # (sumN,3)
    y = torch.tensor(ys, dtype=torch.float32)
    return coords, feats, y


def collate_me_fusion(batch, plane_names=("u", "v", "w")):
    coords_by_plane = {name: [] for name in plane_names}
    feats_by_plane = {name: [] for name in plane_names}
    ys = []

    batch_size = len(batch)
    available_mask = torch.zeros((batch_size, len(plane_names)), dtype=torch.float32)

    for bi, (c, f, y) in enumerate(batch):
        c = torch.as_tensor(c, dtype=torch.int32).contiguous()   # (N,3): (plane,y,x)
        f = torch.as_tensor(f, dtype=torch.float32).contiguous() # (N,3): (occ, logq, plane_id)
        ys.append(y)

        for pi, name in enumerate(plane_names):
            mask = c[:, 0] == pi
            if mask.any():
                available_mask[bi, pi] = 1.0
                c_plane = c[mask][:, 1:3]
                f_plane = f[mask][:, 0:2]
            else:
                c_plane = torch.zeros((1, 2), dtype=torch.int32)
                f_plane = torch.zeros((1, 2), dtype=torch.float32)

            bcol = torch.full((c_plane.shape[0], 1), bi, dtype=torch.int32)
            coords_by_plane[name].append(torch.cat([bcol, c_plane], dim=1))
            feats_by_plane[name].append(f_plane)

    coords = {
        name: torch.cat(coords_by_plane[name], dim=0).contiguous()
        for name in plane_names
    }
    feats = {
        name: torch.cat(feats_by_plane[name], dim=0).contiguous()
        for name in plane_names
    }
    y = torch.tensor(ys, dtype=torch.float32)
    return coords, feats, y, available_mask
