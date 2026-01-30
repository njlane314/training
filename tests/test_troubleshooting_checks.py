import os
import tempfile
import unittest

import numpy as np
import torch

from likelihood.data import pack_events


class TestTroubleshootingChecks(unittest.TestCase):
    def test_placeholder_fraction_and_quantiles(self):
        with tempfile.TemporaryDirectory() as d:
            labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.uint8)
            nnz = np.array([1, 2, 3, 1, 4, 5], dtype=np.int32)

            torch.save(
                {"labels": torch.from_numpy(labels), "nnz": torch.from_numpy(nnz)},
                os.path.join(d, "index.pt"),
            )

            meta = torch.load(os.path.join(d, "index.pt"), map_location="cpu")
            labels_loaded = np.asarray(meta["labels"], dtype=np.uint8)
            nnz_loaded = np.asarray(meta["nnz"], dtype=np.int32)

            placeholder_frac = float(np.mean(nnz_loaded == 1))
            self.assertAlmostEqual(placeholder_frac, 2.0 / 6.0)

            q0 = np.quantile(nnz_loaded[labels_loaded == 0], [0, 0.5, 1.0])
            q1 = np.quantile(nnz_loaded[labels_loaded == 1], [0, 0.5, 1.0])

            np.testing.assert_allclose(q0, np.array([1.0, 3.0, 4.0]))
            np.testing.assert_allclose(q1, np.array([1.0, 2.0, 5.0]))

    def test_index_and_shard_labels_align(self):
        with tempfile.TemporaryDirectory() as d:
            shard_events = 4
            all_labels = []

            for sid in [0, 1]:
                coords_list = []
                feats_list = []
                labels = []
                start_event = sid * shard_events

                for i in range(shard_events):
                    gi = start_event + i
                    coords = np.array([[gi % 3, i, i]], dtype=np.int32)
                    feats = np.array([[float(gi), 0.0, 0.0, 0.0]], dtype=np.float32)
                    coords_list.append(coords)
                    feats_list.append(feats)
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
                {"shard_events": int(shard_events), "labels": torch.tensor(all_labels, dtype=torch.uint8)},
                os.path.join(d, "index.pt"),
            )

            meta = torch.load(os.path.join(d, "index.pt"), map_location="cpu")
            labels_all = np.asarray(meta["labels"], dtype=np.uint8)
            shard_events_meta = int(meta["shard_events"])

            rng = np.random.default_rng(0)
            for _ in range(10):
                gi = int(rng.integers(0, len(labels_all)))
                sid = gi // shard_events_meta
                local = gi - sid * shard_events_meta
                shard = torch.load(os.path.join(d, f"shard_{sid:05d}.pt"), map_location="cpu")
                y_shard = int(shard["labels"][local].item())
                self.assertEqual(y_shard, int(labels_all[gi]))


if __name__ == "__main__":
    unittest.main()
