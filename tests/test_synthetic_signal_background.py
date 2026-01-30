import os
import tempfile
import unittest

import numpy as np
import torch

from likelihood.data import ShardDataset, pack_events, plane_to_sparse


def _build_synthetic_events(n_events=12, hits_per_event=16, H=16, W=16, seed=0):
    rng = np.random.default_rng(seed)
    coords_list = []
    feats_list = []
    labels = []

    for i in range(n_events):
        y = 1 if (i % 2 == 0) else 0
        labels.append(y)

        view = rng.integers(0, 3, size=hits_per_event, dtype=np.int32)
        yy = rng.integers(0, H // 2, size=hits_per_event, dtype=np.int32) * 2
        xx = rng.integers(0, W // 2, size=hits_per_event, dtype=np.int32) * 2

        coords = np.stack([view, yy, xx], axis=1).astype(np.int32)

        adc_base = 2.0 if y == 1 else 0.1
        adc = (adc_base + rng.normal(0.0, 0.01, size=hits_per_event)).astype(np.float32)
        y_norm = (yy.astype(np.float32) - (H / 2.0)) / (H / 2.0)
        x_norm = (xx.astype(np.float32) - (W / 2.0)) / (W / 2.0)
        v_norm = (view.astype(np.float32) - 1.0)
        feats = np.stack([adc, y_norm, x_norm, v_norm], axis=1).astype(np.float32)

        coords_list.append(coords)
        feats_list.append(feats)

    return coords_list, feats_list, np.asarray(labels, dtype=np.uint8)


class TestSyntheticSignalBackground(unittest.TestCase):
    def test_plane_to_sparse_rejects_wrong_shape(self):
        flat = np.zeros(10, dtype=np.float32)
        with self.assertRaises(ValueError):
            plane_to_sparse(flat, view=0, H=2, W=6, thr=0.0, signlog=False)

    def test_signal_background_roundtrip_is_learnable(self):
        coords_list, feats_list, labels = _build_synthetic_events()

        with tempfile.TemporaryDirectory() as d:
            coords_t, feats_t, starts_t = pack_events(coords_list, feats_list, feat_dtype=np.float16)
            torch.save(
                {
                    "start_event": 0,
                    "n_events": int(labels.shape[0]),
                    "coords": coords_t,
                    "feats": feats_t,
                    "starts": starts_t,
                    "labels": torch.from_numpy(labels),
                },
                os.path.join(d, "shard_00000.pt"),
            )
            torch.save(
                {"shard_events": int(labels.shape[0]), "labels": torch.from_numpy(labels)},
                os.path.join(d, "index.pt"),
            )

            ds = ShardDataset(d, np.arange(labels.shape[0], dtype=np.int64))

            adc_means = []
            for i in range(len(ds)):
                coords, feats, y = ds[i]
                np.testing.assert_array_equal(coords.numpy(), coords_list[i])
                np.testing.assert_allclose(feats.numpy(), feats_list[i], atol=1e-3, rtol=0.0)
                self.assertEqual(int(y), int(labels[i]))
                adc_means.append(float(feats[:, 0].mean().item()))

            adc_means = np.asarray(adc_means)
            mean_sig = float(adc_means[labels == 1].mean())
            mean_bkg = float(adc_means[labels == 0].mean())
            self.assertGreater(mean_sig, mean_bkg + 0.5)

            threshold = 0.5 * (mean_sig + mean_bkg)
            preds = (adc_means > threshold).astype(np.uint8)
            acc = float(np.mean(preds == labels))
            self.assertGreaterEqual(acc, 0.95)


if __name__ == "__main__":
    unittest.main()
