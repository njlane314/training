import os
import tempfile
import unittest

import numpy as np
import torch

from likelihood.data import ShardDataset, pack_events


class TestShardsRoundTrip(unittest.TestCase):
    def test_dataset_reads(self):
        with tempfile.TemporaryDirectory() as d:
            labels = torch.tensor([0, 1, 0, 1], dtype=torch.uint8)
            meta = {"H": 4, "W": 4, "shard_events": 4, "n_events": 4, "labels": labels}
            torch.save(meta, os.path.join(d, "index.pt"))

            coords_list = [
                np.array([[0, 0, 0]], dtype=np.int32),
                np.array([[1, 1, 1]], dtype=np.int32),
                np.array([[2, 2, 2]], dtype=np.int32),
                np.array([[0, 3, 3]], dtype=np.int32),
            ]
            feats_list = [
                np.array([[0.1, -1.0, -1.0, -1.0]], dtype=np.float32),
                np.array([[0.2, -0.5, -0.5, 0.0]], dtype=np.float32),
                np.array([[0.3, 0.0, 0.0, 1.0]], dtype=np.float32),
                np.array([[0.4, 0.5, 0.5, -1.0]], dtype=np.float32),
            ]
            coords_t, feats_t, starts_t = pack_events(coords_list, feats_list, feat_dtype=np.float16)
            torch.save(
                {"start_event": 0, "n_events": 4, "coords": coords_t, "feats": feats_t, "starts": starts_t},
                os.path.join(d, "shard_00000.pt"),
            )

            ds = ShardDataset(d, np.array([0, 1, 2, 3], dtype=np.int64))
            c, f, y = ds[2]
            self.assertEqual(int(y), 0)
            self.assertEqual(c.shape, (1, 3))
            self.assertEqual(f.shape, (1, 4))
            self.assertEqual(f.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
