import importlib.util
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

HAS_ME = importlib.util.find_spec("MinkowskiEngine") is not None


@unittest.skipUnless(HAS_ME, "requires MinkowskiEngine")
class TestTrainStepMechanics(unittest.TestCase):
    def test_grad_accumulation_changes_weights_only_on_step(self):
        import MinkowskiEngine as ME
        from likelihood.data import collate
        from likelihood.model import MinkUNetClassifier

        c0 = np.array([[0, 0, 0]], dtype=np.int32)
        f0 = np.array([[1.0, -1.0, -1.0, -1.0]], dtype=np.float32)
        coords, feats, y = collate([(c0, f0, 1.0), (c0, f0, 0.0)])
        x = ME.SparseTensor(feats, coords, device="cpu")
        y = y.float().view(-1)

        model = MinkUNetClassifier(in_channels=4, base=8, strides=0, dropout=0.0).train()
        opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
        loss_fn = nn.BCEWithLogitsLoss()

        w0 = model.head[-1].weight.detach().clone()

        opt.zero_grad(set_to_none=True)

        logits = model(x).view(-1)
        loss = loss_fn(logits, y)
        loss.backward()
        w1 = model.head[-1].weight.detach().clone()
        self.assertTrue(torch.allclose(w0, w1), "weights changed before optimizer step")

        logits2 = model(x).view(-1)
        loss2 = loss_fn(logits2, y)
        loss2.backward()

        opt.step()
        w2 = model.head[-1].weight.detach().clone()
        self.assertFalse(torch.allclose(w0, w2), "weights did not change after optimizer step")


if __name__ == "__main__":
    unittest.main()
