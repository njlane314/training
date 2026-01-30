import os
import tempfile
import unittest

import numpy as np
import torch

from likelihood.data import pack_events, ShardDataset


def _make_event(coords, feats):
    return np.asarray(coords, dtype=np.int32), np.asarray(feats, dtype=np.float32)


class TestPackEvents(unittest.TestCase):
    def test_starts_and_roundtrip_slicing(self):
        c0, f0 = _make_event([[0, 0, 0], [0, 1, 1]], [[1, 0, 0, -1], [2, 0, 0, -1]])
        c1, f1 = _make_event([[2, 3, 3]], [[3, 0, 0, 1]])

        coords_t, feats_t, starts_t = pack_events([c0, c1], [f0, f1], feat_dtype=np.float16)

        self.assertEqual(coords_t.dtype, torch.int32)
        self.assertEqual(feats_t.dtype, torch.float16)
        self.assertEqual(starts_t.dtype, torch.int64)

        self.assertTrue(torch.equal(starts_t, torch.tensor([0, 2, 3], dtype=torch.int64)))

        s0, e0 = int(starts_t[0]), int(starts_t[1])
        s1, e1 = int(starts_t[1]), int(starts_t[2])

        self.assertTrue(torch.equal(coords_t[s0:e0], torch.tensor(c0, dtype=torch.int32)))
        self.assertTrue(torch.equal(coords_t[s1:e1], torch.tensor(c1, dtype=torch.int32)))


class TestShardDataset(unittest.TestCase):
    def test_getitem_and_getitems_match(self):
        with tempfile.TemporaryDirectory() as d:
            shard_events = 3
            all_labels = []

            for sid in [0, 1]:
                coords_list = []
                feats_list = []
                labels = []
                start_event = sid * shard_events

                for i in range(shard_events):
                    gi = start_event + i
                    c = np.array([[gi % 3, i, i]], dtype=np.int32)
                    f = np.array(
                        [[float(gi + 1), 0.0, 0.0, float((gi % 3) - 1)]],
                        dtype=np.float32,
                    )
                    coords_list.append(c)
                    feats_list.append(f)
                    y = 1 if (gi % 2 == 0) else 0
                    labels.append(y)
                    all_labels.append(y)

                coords_t, feats_t, starts_t = pack_events(coords_list, feats_list, feat_dtype=np.float16)
                torch.save(
                    {
                        "start_event": int(start_event),
                        "n_events": int(shard_events),
                        "coords": coords_t,
                        "feats": feats_t,
                        "starts": starts_t,
                        "labels": torch.tensor(labels, dtype=torch.uint8),
                    },
                    os.path.join(d, f"shard_{sid:05d}.pt"),
                )

            torch.save(
                {
                    "shard_events": int(shard_events),
                    "labels": torch.tensor(all_labels, dtype=torch.uint8),
                },
                os.path.join(d, "index.pt"),
            )

            event_indices = np.array([0, 2, 3, 5], dtype=np.int64)
            ds = ShardDataset(d, event_indices, cache_size=1)

            c0, f0, y0 = ds[0]
            self.assertEqual(float(y0), float(all_labels[0]))
            self.assertTrue(torch.equal(c0, torch.tensor([[0, 0, 0]], dtype=torch.int32)))

            idxs = np.array([0, 1, 2, 3], dtype=np.int64)
            batch = ds.__getitems__(idxs)
            for j, i in enumerate(idxs):
                c_a, f_a, y_a = batch[j]
                c_b, f_b, y_b = ds[int(i)]
                self.assertTrue(torch.equal(c_a, c_b))
                self.assertTrue(torch.allclose(f_a, f_b))
                self.assertEqual(float(y_a), float(y_b))

    def test_cache_eviction(self):
        with tempfile.TemporaryDirectory() as d:
            shard_events = 2
            all_labels = [0, 1, 0, 1]

            c0, f0 = np.array([[0, 0, 0]], np.int32), np.array([[1, 0, 0, -1]], np.float32)
            c1, f1 = np.array([[1, 0, 1]], np.int32), np.array([[2, 0, 0, 0]], np.float32)
            coords_t, feats_t, starts_t = pack_events([c0, c1], [f0, f1])
            torch.save(
                {
                    "start_event": 0,
                    "n_events": 2,
                    "coords": coords_t,
                    "feats": feats_t,
                    "starts": starts_t,
                    "labels": torch.tensor([0, 1], dtype=torch.uint8),
                },
                os.path.join(d, "shard_00000.pt"),
            )

            c2, f2 = np.array([[2, 1, 0]], np.int32), np.array([[3, 0, 0, 1]], np.float32)
            c3, f3 = np.array([[0, 1, 1]], np.int32), np.array([[4, 0, 0, -1]], np.float32)
            coords_t, feats_t, starts_t = pack_events([c2, c3], [f2, f3])
            torch.save(
                {
                    "start_event": 2,
                    "n_events": 2,
                    "coords": coords_t,
                    "feats": feats_t,
                    "starts": starts_t,
                    "labels": torch.tensor([0, 1], dtype=torch.uint8),
                },
                os.path.join(d, "shard_00001.pt"),
            )

            torch.save(
                {"shard_events": shard_events, "labels": torch.tensor(all_labels, dtype=torch.uint8)},
                os.path.join(d, "index.pt"),
            )

            ds = ShardDataset(d, np.array([0, 1, 2, 3], dtype=np.int64), cache_size=1)

            _ = ds[0]
            self.assertEqual(list(ds._cache.keys()), [0])
            _ = ds[2]
            self.assertEqual(list(ds._cache.keys()), [1])
            _ = ds[1]
            self.assertEqual(list(ds._cache.keys()), [0])


if __name__ == "__main__":
    unittest.main()
