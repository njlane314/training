import importlib.util
import unittest

import numpy as np
import torch

HAS_ME = importlib.util.find_spec("MinkowskiEngine") is not None


@unittest.skipUnless(HAS_ME, "requires MinkowskiEngine")
class TestMinkowskiAssumptions(unittest.TestCase):
    def test_global_pooling_is_average(self):
        """
        Assumption: ME.MinkowskiGlobalPooling() is GLOBAL_AVG, not GLOBAL_SUM.
        If this fails, make pooling explicit in your model (set the mode).
        """
        import MinkowskiEngine as ME

        coords = torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 2],
            ],
            dtype=torch.int32,
        )
        feats = torch.tensor([[1.0], [3.0]], dtype=torch.float32)
        x = ME.SparseTensor(feats, coords, device="cpu")

        pool = ME.MinkowskiGlobalPooling()
        y = pool(x)

        got = float(y.F[0, 0].item())
        # If AVG: (1+3)/2 = 2. If SUM: 4.
        self.assertAlmostEqual(
            got,
            2.0,
            places=6,
            msg=(
                f"MinkowskiGlobalPooling() is not behaving as AVG (got={got}). "
                "If your analysis assumes AVG pooling, set it explicitly in code."
            ),
        )

    def test_residual_block_preserves_coordinate_set(self):
        """
        Assumption: your ResidualBlock uses submanifold-like behavior (same coords in/out).
        If this fails, you are expanding support inside the residual branch.
        """
        import MinkowskiEngine as ME
        from likelihood.model import ResidualBlock

        torch.manual_seed(0)

        # 1 batch, D=3 => coords have 4 cols: (batch, view, y, x)
        coords = torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 1, 2, 4],
                [0, 2, 4, 6],
            ],
            dtype=torch.int32,
        )
        feats = torch.randn(coords.shape[0], 4, dtype=torch.float32)
        x = ME.SparseTensor(feats, coords, device="cpu")

        blk = ResidualBlock(in_ch=4, out_ch=4, dim=3).eval()
        y = blk(x)

        self.assertTrue(
            torch.equal(y.C, x.C),
            msg=(
                "ResidualBlock changed the coordinate set. "
                "If you assume coordinate-preserving (submanifold) convs inside residuals, "
                "switch to the correct ME layer/config that guarantees same support."
            ),
        )

    def test_model_has_cross_view_kernel_somewhere(self):
        """
        Assumption: treating 'view' as a spatial dimension is only meaningful if
        at least one conv has kernel_size[0] > 1 (can couple adjacent views).
        """
        import MinkowskiEngine as ME
        from likelihood.model import MinkUNetClassifier

        model = MinkUNetClassifier(in_channels=4, base=8, strides=1, dropout=0.0).eval()

        def as_tuple(x):
            if x is None:
                return None
            if isinstance(x, int):
                return (x,)
            try:
                return tuple(int(v) for v in x)
            except Exception:
                return None

        found = False
        for m in model.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ks = as_tuple(getattr(m, "kernel_size", None))
                if ks is not None and len(ks) >= 1 and ks[0] > 1:
                    found = True
                    break

        self.assertTrue(
            found,
            msg=(
                "No MinkowskiConvolution with kernel_size along view > 1 was found. "
                "If you intend cross-view mixing, ensure at least one conv uses ks_view > 1."
            ),
        )

    def test_unet_skip_alignment_after_upsample(self):
        """
        Assumption: after downsample then transpose-conv upsample, coordinates match
        the corresponding skip tensor so ME.cat(...) is well-defined.
        """
        import MinkowskiEngine as ME
        from likelihood.model import MinkUNetClassifier

        torch.manual_seed(0)
        rng = np.random.default_rng(0)

        model = MinkUNetClassifier(in_channels=4, base=8, strides=1, dropout=0.0).eval()

        # Build coords that are EVEN in (y,x) so stride-2 down/up can invert cleanly.
        n = 32
        view = rng.integers(0, 3, size=n, dtype=np.int32)
        y = rng.integers(0, 16, size=n, dtype=np.int32) * 2
        x = rng.integers(0, 16, size=n, dtype=np.int32) * 2

        coords = np.stack([np.zeros(n, dtype=np.int32), view, y, x], axis=1)  # batch=0
        coords_t = torch.from_numpy(coords).to(torch.int32)

        # Features: adc, y_norm, x_norm, v_norm
        H = W = 32
        y_norm = (y.astype(np.float32) - (H / 2.0)) / (H / 2.0)
        x_norm = (x.astype(np.float32) - (W / 2.0)) / (W / 2.0)
        v_norm = (view.astype(np.float32) - 1.0)
        adc = np.ones(n, dtype=np.float32)
        feats = np.stack([adc, y_norm, x_norm, v_norm], axis=1).astype(np.float32)
        feats_t = torch.from_numpy(feats)

        x0 = ME.SparseTensor(feats_t, coords_t, device="cpu")

        # Manual forward through the first encoder level, downsample, mid, then first upsample.
        with torch.no_grad():
            x = model.inorm(x0)
            x = model.c0(x)
            x = model.enc[0](x)
            skip = x
            x = model.enc[1](x)  # stride (1,2,2)
            x = model.mid(x)
            x_up = model.dec[0](x)  # transpose stride (1,2,2)

        self.assertTrue(
            torch.equal(x_up.C, skip.C),
            msg=(
                "Upsampled coords do not match skip coords. If ME.cat(x_up, skip) "
                "is assumed to concatenate aligned features, this mismatch will break the U-Net logic."
            ),
        )

    def test_forward_is_deterministic_in_eval_mode(self):
        import MinkowskiEngine as ME
        from likelihood.model import MinkUNetClassifier

        torch.manual_seed(0)

        coords = torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 1, 2, 2],
                [0, 2, 4, 4],
            ],
            dtype=torch.int32,
        )
        feats = torch.tensor(
            [
                [1.0, -1.0, -1.0, -1.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.5, 0.5, 1.0],
            ],
            dtype=torch.float32,
        )
        x = ME.SparseTensor(feats, coords, device="cpu")

        model = MinkUNetClassifier(in_channels=4, base=8, strides=1, dropout=0.0).eval()
        with torch.no_grad():
            a = model(x).clone()
            b = model(x).clone()

        self.assertTrue(torch.allclose(a, b, atol=0.0, rtol=0.0))


if __name__ == "__main__":
    unittest.main()
