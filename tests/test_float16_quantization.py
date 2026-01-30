import unittest

import numpy as np
import torch

from likelihood.data import ShardDataset, pack_events


class TestFloat16Packing(unittest.TestCase):
    def test_pack_events_float16_has_reasonable_error(self):
        rng = np.random.default_rng(0)

        coords_list = []
        feats_list = []

        for _ in range(3):
            n = 50
            coords = rng.integers(0, 16, size=(n, 3), dtype=np.int32)  # (view,y,x)-like
            feats = rng.uniform(-3.0, 3.0, size=(n, 4)).astype(np.float32)
            coords_list.append(coords)
            feats_list.append(feats)

        coords_t, feats_t, starts_t = pack_events(coords_list, feats_list, feat_dtype=np.float16)

        # Roundtrip compare per-event segments.
        feats_rt = feats_t.to(torch.float32).numpy()
        starts = starts_t.numpy()

        max_abs = 0.0
        for i in range(len(coords_list)):
            s, e = int(starts[i]), int(starts[i + 1])
            orig = feats_list[i]
            rt = feats_rt[s:e]
            max_abs = max(max_abs, float(np.max(np.abs(orig - rt))))

        # float16 is coarse; but for values O(1), <=~1e-2 abs error is a reasonable expectation.
        self.assertLess(
            max_abs,
            5e-2,
            msg=f"float16 packing error too large: max_abs={max_abs}",
        )

    def test_sharddataset_slice_returns_float32_features(self):
        # Minimal fake shard dict to test ShardDataset._slice_one
        d = {
            "starts": torch.tensor([0, 2], dtype=torch.int64),
            "coords": torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.int32),
            "feats": torch.tensor(
                [[1.0, 2.0, 3.0, 4.0], [0.5, -0.5, 0.25, -0.25]], dtype=torch.float16
            ),
        }
        c, f = ShardDataset._slice_one(d, local=0)
        self.assertEqual(c.dtype, torch.int32)
        self.assertEqual(f.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
