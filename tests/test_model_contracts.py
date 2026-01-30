import importlib.util
import unittest

import numpy as np
import torch
import torch.nn as nn

HAS_ME = importlib.util.find_spec("MinkowskiEngine") is not None


@unittest.skipUnless(HAS_ME, "requires MinkowskiEngine")
class TestModelContracts(unittest.TestCase):
    def _make_sparse_batch(self):
        import MinkowskiEngine as ME
        from likelihood.data import collate

        c0 = np.array([[0, 0, 0]], dtype=np.int32)
        f0 = np.array([[1.0, -1.0, -1.0, -1.0]], dtype=np.float32)

        c1 = np.array([[2, 1, 1], [2, 1, 2]], dtype=np.int32)
        f1 = np.array([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]], dtype=np.float32)

        coords, feats, y = collate([(c0, f0, 1.0), (c1, f1, 0.0)])
        x = ME.SparseTensor(feats, coords, device="cpu")
        return x, y

    def test_forward_shape_and_finiteness(self):
        from likelihood.model import MinkUNetClassifier

        x, y = self._make_sparse_batch()
        model = MinkUNetClassifier(in_channels=4, base=8, strides=1, dropout=0.0).eval()
        with torch.no_grad():
            out = model(x)
        self.assertEqual(tuple(out.shape), (2, 1))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_backward_produces_gradients(self):
        from likelihood.model import MinkUNetClassifier

        x, y = self._make_sparse_batch()
        model = MinkUNetClassifier(in_channels=4, base=8, strides=1, dropout=0.0).train()
        loss_fn = nn.BCEWithLogitsLoss()

        logits = model(x).view(-1)
        loss = loss_fn(logits, y.float().view(-1))
        loss.backward()

        g = model.head[-1].weight.grad
        self.assertIsNotNone(g)
        self.assertTrue(torch.isfinite(g).all().item())
        self.assertGreater(float(g.abs().sum().item()), 0.0)

    def test_log_count_feature_is_correct_last_column(self):
        from likelihood.model import MinkUNetClassifier

        x, y = self._make_sparse_batch()
        model = MinkUNetClassifier(in_channels=4, base=8, strides=0, dropout=0.0).eval()

        model.head = nn.Identity()

        with torch.no_grad():
            z = model(x)
        self.assertEqual(z.shape[0], 2)
        self.assertEqual(z.shape[1], 2 * 8 + 1)

        expected = torch.log1p(torch.tensor([1.0, 2.0]))
        got = z[:, -1].cpu()
        self.assertTrue(torch.allclose(got, expected, atol=1e-6, rtol=0.0))

    def test_anisotropic_striding_in_view_axis(self):
        import MinkowskiEngine as ME
        from likelihood.model import MinkUNetClassifier

        model = MinkUNetClassifier(in_channels=4, base=8, strides=2, dropout=0.0)

        def _as_tuple(v):
            if isinstance(v, int):
                return (v,)
            try:
                return tuple(int(x) for x in v)
            except Exception:
                return None

        for m in model.modules():
            if isinstance(m, ME.MinkowskiConvolution) or isinstance(m, ME.MinkowskiConvolutionTranspose):
                st = _as_tuple(getattr(m, "stride", None))
                if st is None:
                    continue
                if any(s != 1 for s in st):
                    self.assertEqual(st[0], 1, msg=f"strided layer strides in view axis: stride={st}")


if __name__ == "__main__":
    unittest.main()
