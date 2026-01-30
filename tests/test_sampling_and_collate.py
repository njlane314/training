import unittest

import numpy as np
import torch

from likelihood.data import BalancedBatchSampler, collate


class TestBalancedBatchSampler(unittest.TestCase):
    def test_requires_even_batch_size(self):
        labels = np.array([0, 1, 0, 1], dtype=np.uint8)
        shard_ids = np.array([0, 0, 1, 1], dtype=np.int64)
        local_ids = np.array([0, 1, 0, 1], dtype=np.int64)
        with self.assertRaises(ValueError):
            BalancedBatchSampler(labels, shard_ids, local_ids, batch_size=3, seed=1)

    def test_batches_are_balanced_and_sorted_for_io(self):
        labels = np.array([1, 0] * 6, dtype=np.uint8)

        shard_ids = np.array([1, 0, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64)
        local_ids = np.array([5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 10, 11], dtype=np.int64)

        bs = 4
        sampler = BalancedBatchSampler(labels, shard_ids, local_ids, batch_size=bs, seed=123, steps=5)
        sampler.set_epoch(0)

        for batch in sampler:
            self.assertEqual(len(batch), bs)
            batch = np.asarray(batch, dtype=np.int64)
            y = labels[batch]
            self.assertEqual(int((y == 1).sum()), bs // 2)
            self.assertEqual(int((y == 0).sum()), bs // 2)

            keys = np.stack([shard_ids[batch], local_ids[batch]], axis=1)
            for i in range(1, keys.shape[0]):
                self.assertTrue(
                    (keys[i - 1, 0] < keys[i, 0])
                    or (keys[i - 1, 0] == keys[i, 0] and keys[i - 1, 1] <= keys[i, 1])
                )

    def test_deterministic_per_epoch(self):
        labels = np.array([1, 0] * 50, dtype=np.uint8)
        shard_ids = np.zeros_like(labels, dtype=np.int64)
        local_ids = np.arange(labels.size, dtype=np.int64)

        s1 = BalancedBatchSampler(labels, shard_ids, local_ids, batch_size=10, seed=7, steps=3)
        s2 = BalancedBatchSampler(labels, shard_ids, local_ids, batch_size=10, seed=7, steps=3)

        s1.set_epoch(0)
        s2.set_epoch(0)
        self.assertEqual(list(iter(s1)), list(iter(s2)))

        s1.set_epoch(1)
        s2.set_epoch(1)
        self.assertEqual(list(iter(s1)), list(iter(s2)))


class TestCollate(unittest.TestCase):
    def test_collate_adds_batch_index(self):
        c0 = np.array([[0, 0, 0], [1, 0, 1]], dtype=np.int32)
        f0 = np.array([[1, 0, 0, -1], [2, 0, 0, 0]], dtype=np.float32)
        y0 = 1.0

        c1 = np.array([[2, 1, 1]], dtype=np.int32)
        f1 = np.array([[3, 0, 0, 1]], dtype=np.float32)
        y1 = 0.0

        coords, feats, y = collate([(c0, f0, y0), (c1, f1, y1)])

        self.assertEqual(coords.shape[1], 4)
        self.assertEqual(feats.shape[1], 4)
        self.assertEqual(y.shape[0], 2)

        self.assertTrue((coords[:2, 0] == 0).all())
        self.assertTrue((coords[2:, 0] == 1).all())

    def test_collate_handles_coords_without_view(self):
        c0 = np.array([[1, 2]], dtype=np.int32)
        f0 = np.array([[1, 0, 0, 0]], dtype=np.float32)
        coords, feats, y = collate([(c0, f0, 1.0)])
        self.assertEqual(coords.shape, (1, 4))
        self.assertEqual(int(coords[0, 1]), 0)
        self.assertEqual(int(coords[0, 2]), 1)
        self.assertEqual(int(coords[0, 3]), 2)

    def test_collate_handles_coords_with_existing_batch_col(self):
        c0 = np.array([[99, 2, 3, 4]], dtype=np.int32)
        f0 = np.array([[1, 0, 0, 1]], dtype=np.float32)
        coords, feats, y = collate([(c0, f0, 0.0)])
        self.assertEqual(int(coords[0, 0]), 0)
        self.assertEqual(int(coords[0, 1]), 2)
        self.assertEqual(int(coords[0, 2]), 3)
        self.assertEqual(int(coords[0, 3]), 4)


if __name__ == "__main__":
    unittest.main()
