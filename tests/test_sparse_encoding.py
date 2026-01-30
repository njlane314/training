import math
import unittest

import numpy as np

from likelihood.data import plane_to_sparse, event_to_sparse


class TestPlaneToSparse(unittest.TestCase):
    def test_rejects_wrong_plane_size(self):
        H, W = 4, 4
        flat = np.zeros(H * W - 1, dtype=np.float32)
        with self.assertRaises(ValueError):
            plane_to_sparse(flat, view=0, H=H, W=W, thr=0.0, signlog=False)

    def test_empty_plane_returns_none(self):
        H, W = 4, 4
        flat = np.zeros(H * W, dtype=np.float32)
        coords, feats = plane_to_sparse(flat, view=1, H=H, W=W, thr=0.0, signlog=False)
        self.assertIsNone(coords)
        self.assertIsNone(feats)

    def test_relu_log_features_matches_expected(self):
        H, W = 4, 4
        flat = np.zeros(H * W, dtype=np.float32)
        flat[0] = 1.0
        flat[5] = 3.0

        coords, feats = plane_to_sparse(flat, view=2, H=H, W=W, thr=0.0, signlog=False)
        self.assertEqual(coords.shape, (2, 3))
        self.assertEqual(feats.shape, (2, 4))

        self.assertTrue((coords[:, 0] == 2).all())
        self.assertTrue((coords[:, 1] == np.array([0, 1], dtype=np.int32)).all())
        self.assertTrue((coords[:, 2] == np.array([0, 1], dtype=np.int32)).all())

        self.assertAlmostEqual(float(feats[0, 0]), math.log(2.0), places=6)
        self.assertAlmostEqual(float(feats[1, 0]), math.log(4.0), places=6)

        self.assertAlmostEqual(float(feats[0, 1]), -1.0, places=6)
        self.assertAlmostEqual(float(feats[0, 2]), -1.0, places=6)
        self.assertAlmostEqual(float(feats[0, 3]), 1.0, places=6)

        self.assertAlmostEqual(float(feats[1, 1]), -0.5, places=6)
        self.assertAlmostEqual(float(feats[1, 2]), -0.5, places=6)
        self.assertAlmostEqual(float(feats[1, 3]), 1.0, places=6)

    def test_threshold_behavior_relu_log(self):
        H, W = 4, 4
        flat = np.zeros(H * W, dtype=np.float32)
        flat[0] = 0.1
        flat[1] = 0.2
        flat[2] = 0.3

        coords, feats = plane_to_sparse(flat, view=0, H=H, W=W, thr=0.2, signlog=False)
        self.assertEqual(coords.shape[0], 1)
        self.assertEqual(int(coords[0, 1]), 0)
        self.assertEqual(int(coords[0, 2]), 2)
        self.assertAlmostEqual(float(feats[0, 0]), math.log1p(0.3), places=6)

    def test_signlog_includes_negative_values(self):
        H, W = 4, 4
        flat = np.zeros(H * W, dtype=np.float32)
        flat[6] = -3.0

        coords, feats = plane_to_sparse(flat, view=1, H=H, W=W, thr=0.0, signlog=True)
        self.assertEqual(coords.shape, (1, 3))
        self.assertEqual(feats.shape, (1, 4))
        self.assertAlmostEqual(float(feats[0, 0]), -math.log1p(3.0), places=6)

        coords2, feats2 = plane_to_sparse(flat, view=1, H=H, W=W, thr=0.0, signlog=False)
        self.assertIsNone(coords2)
        self.assertIsNone(feats2)

    def test_normalisation_edges(self):
        H, W = 4, 4
        flat = np.zeros(H * W, dtype=np.float32)
        flat[15] = 1.0
        coords, feats = plane_to_sparse(flat, view=0, H=H, W=W, thr=0.0, signlog=False)
        self.assertEqual(coords.tolist(), [[0, 3, 3]])
        self.assertAlmostEqual(float(feats[0, 1]), (3.0 - 2.0) / 2.0, places=6)
        self.assertAlmostEqual(float(feats[0, 2]), (3.0 - 2.0) / 2.0, places=6)
        self.assertAlmostEqual(float(feats[0, 3]), -1.0, places=6)


class TestEventToSparse(unittest.TestCase):
    def test_merges_views_and_preserves_view_coord(self):
        H, W = 4, 4
        u = np.zeros(H * W, dtype=np.float32)
        v = np.zeros(H * W, dtype=np.float32)
        w = np.zeros(H * W, dtype=np.float32)

        u[0] = 1.0
        w[5] = 2.0

        coords, feats = event_to_sparse(u, v, w, H=H, W=W, thr=0.0, signlog=False)
        self.assertEqual(coords.shape[1], 3)
        self.assertEqual(feats.shape[1], 4)
        self.assertEqual(coords.shape[0], 2)

        self.assertEqual(int(coords[0, 0]), 0)
        self.assertEqual(int(coords[1, 0]), 2)

    def test_empty_event_returns_dummy_site(self):
        H, W = 4, 4
        z = np.zeros(H * W, dtype=np.float32)
        coords, feats = event_to_sparse(z, z, z, H=H, W=W, thr=0.0, signlog=False)
        self.assertEqual(coords.shape, (1, 3))
        self.assertEqual(feats.shape, (1, 4))
        self.assertTrue((coords == np.array([[0, 0, 0]], dtype=np.int32)).all())
        self.assertTrue((feats == 0).all())


if __name__ == "__main__":
    unittest.main()
