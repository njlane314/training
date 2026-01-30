import importlib.util
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from likelihood.data import collate
from likelihood.model import MinkUNetClassifier

HAS_ME = importlib.util.find_spec("MinkowskiEngine") is not None


class ToySparseDataset(torch.utils.data.Dataset):
    """
    In-memory toy dataset producing (coords(view,y,x), feats(4), label).
    Designed to be *very* easy: signal has higher adc everywhere than background.
    Coordinates are even in (y,x) so stride-2 down/up can align cleanly.
    """

    def __init__(self, n_events=64, hits_per_event=32, H=32, W=32, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.H = int(H)
        self.W = int(W)

        coords_list = []
        feats_list = []
        labels = []

        for i in range(n_events):
            y = 1 if (i % 2 == 0) else 0  # balanced labels (start with signal)
            labels.append(float(y))

            # Same hit count for both classes so log_cnt can't solve it alone.
            n = hits_per_event

            view = rng.integers(0, 3, size=n, dtype=np.int32)
            yy = rng.integers(0, self.H // 2, size=n, dtype=np.int32) * 2  # even y
            xx = rng.integers(0, self.W // 2, size=n, dtype=np.int32) * 2  # even x

            coords = np.stack([view, yy, xx], axis=1).astype(np.int32)

            y_norm = (yy.astype(np.float32) - (self.H / 2.0)) / (self.H / 2.0)
            x_norm = (xx.astype(np.float32) - (self.W / 2.0)) / (self.W / 2.0)
            v_norm = (view.astype(np.float32) - 1.0)

            # adc feature: signal is much larger than background
            adc_base = 2.0 if y == 1 else 0.05
            adc = (adc_base + rng.normal(0.0, 0.01, size=n)).astype(np.float32)

            feats = np.stack([adc, y_norm, x_norm, v_norm], axis=1).astype(np.float32)

            coords_list.append(coords)
            feats_list.append(feats)

        self._coords = coords_list
        self._feats = feats_list
        self._labels = labels

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, i):
        return self._coords[i], self._feats[i], self._labels[i]


@unittest.skipUnless(HAS_ME, "requires MinkowskiEngine")
class TestIntegrationTrainCPU(unittest.TestCase):
    def test_loss_decreases_over_20_steps(self):
        import MinkowskiEngine as ME

        torch.manual_seed(0)
        np.random.seed(0)

        ds = ToySparseDataset(n_events=64, hits_per_event=32, H=32, W=32, seed=0)

        # Deterministic loaders: eval loader no shuffle; train loader shuffle with fixed generator.
        eval_loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate)

        g = torch.Generator().manual_seed(0)
        train_loader = torch.utils.data.DataLoader(
            ds, batch_size=16, shuffle=True, generator=g, collate_fn=collate
        )

        model = MinkUNetClassifier(in_channels=4, base=8, strides=1, dropout=0.0).to("cpu")
        opt = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
        loss_fn = nn.BCEWithLogitsLoss()

        def full_dataset_loss():
            model.eval()
            tot = 0.0
            n = 0
            with torch.no_grad():
                for coords, feats, y in eval_loader:
                    x = ME.SparseTensor(feats, coords, device="cpu")
                    y = y.float().view(-1)
                    logits = model(x).view(-1)
                    loss = loss_fn(logits, y)
                    tot += float(loss.item()) * int(y.numel())
                    n += int(y.numel())
            return tot / max(1, n)

        loss0 = full_dataset_loss()

        model.train()
        it = iter(train_loader)
        for _step in range(20):
            try:
                coords, feats, y = next(it)
            except StopIteration:
                it = iter(train_loader)
                coords, feats, y = next(it)

            x = ME.SparseTensor(feats, coords, device="cpu")
            y = y.float().view(-1)

            logits = model(x).view(-1)
            loss = loss_fn(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        loss1 = full_dataset_loss()

        # Expect a clear reduction from ~0.69 toward something smaller.
        self.assertLess(
            loss1,
            loss0 - 0.10,
            msg=f"Expected loss to decrease by >=0.10. Got loss0={loss0:.4f}, loss1={loss1:.4f}",
        )


if __name__ == "__main__":
    unittest.main()
