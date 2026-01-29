import math
import unittest

import numpy as np

from likelihood.data import plane_to_sparse


class TestFeatures(unittest.TestCase):
    """
    @brief Unit tests for feature extraction utilities.
    """

    def test_relu_log_features(self):
        """
        @brief Validate relu-log feature encoding in plane_to_sparse.
        """
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


if __name__ == "__main__":
    unittest.main()
