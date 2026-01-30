import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
import uproot


class TestWriteShardsFromRootSubprocess(unittest.TestCase):
    def test_write_shards_from_synthetic_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root_path = os.path.join(tmp, "events.root")
            shards_out = os.path.join(tmp, "shards")
            os.makedirs(shards_out, exist_ok=True)

            H = W = 8
            n = 6

            u = np.zeros((n, H * W), dtype=np.float32)
            v = np.zeros_like(u)
            w = np.zeros_like(u)

            # Event patterns:
            # 0: empty
            # 1: U(0,0)=1
            # 2: V(2,4)=2
            # 3: W(0,7)=3
            # 4: U(1,7)=4 and W(1,0)=5
            # 5: empty
            u[1, 0] = 1.0
            v[2, 2 * W + 4] = 2.0
            w[3, 7] = 3.0
            u[4, 1 * W + 7] = 4.0
            w[4, 1 * W + 0] = 5.0

            labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.uint8)
            weights = np.ones(n, dtype=np.float32)

            with uproot.recreate(root_path) as f:
                f.mktree(
                    "events",
                    {
                        "is_signal": "uint8",
                        "w_nominal": "float32",
                        "detector_image_u": f"float32[{H * W}]",
                        "detector_image_v": f"float32[{H * W}]",
                        "detector_image_w": f"float32[{H * W}]",
                    },
                )
                f["events"].extend(
                    {
                        "is_signal": labels,
                        "w_nominal": weights,
                        "detector_image_u": u,
                        "detector_image_v": v,
                        "detector_image_w": w,
                    }
                )

            env = os.environ.copy()
            env.update(
                {
                    "ROOT_FILE": root_path,
                    "TREE": "events",
                    "SHARDS_DIR": shards_out,
                    "SHARDS_OUT": shards_out,
                    "H": str(H),
                    "W": str(W),
                    "THRESH": "0.0",
                    "ADC_SIGNLOG": "0",
                    "SHARD_EVENTS": "4",
                    "CHUNK_EVENTS": "3",
                    "UPROOT_DECOMP_WORKERS": "0",
                }
            )

            code = "from likelihood.data import write_shards_from_root; write_shards_from_root()"
            r = subprocess.run(
                [sys.executable, "-c", code],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                r.returncode,
                0,
                msg=f"subprocess failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}",
            )

            idx = torch.load(os.path.join(shards_out, "index.pt"), map_location="cpu")
            self.assertEqual(int(idx["n_events"]), n)
            self.assertEqual(int(idx["H"]), H)
            self.assertEqual(int(idx["W"]), W)
            self.assertEqual(int(idx["shard_events"]), 4)

            self.assertTrue(torch.equal(idx["labels"].to(torch.uint8), torch.from_numpy(labels)))
            self.assertTrue(torch.allclose(idx["weights"].to(torch.float32), torch.from_numpy(weights)))

            # NOTE: Your current code encodes empty events as a 1-site dummy tensor.
            expected_nnz = torch.tensor([1, 1, 1, 1, 2, 1], dtype=torch.int32)
            self.assertTrue(torch.equal(idx["nnz"].to(torch.int32), expected_nnz))

            shard0 = torch.load(os.path.join(shards_out, "shard_00000.pt"), map_location="cpu")
            shard1 = torch.load(os.path.join(shards_out, "shard_00001.pt"), map_location="cpu")

            self.assertEqual(int(shard0["start_event"]), 0)
            self.assertEqual(int(shard0["n_events"]), 4)
            self.assertEqual(int(shard1["start_event"]), 4)
            self.assertEqual(int(shard1["n_events"]), 2)

            for shard in (shard0, shard1):
                starts = shard["starts"]
                self.assertEqual(starts.dtype, torch.int64)
                self.assertEqual(shard["coords"].dtype, torch.int32)
                self.assertEqual(shard["feats"].dtype, torch.float16)
                self.assertEqual(int(starts[0].item()), 0)
                self.assertEqual(int(starts[-1].item()), int(shard["coords"].shape[0]))
                self.assertEqual(int(starts[-1].item()), int(shard["feats"].shape[0]))
                self.assertEqual(int(starts.numel()), int(shard["n_events"]) + 1)

            # Check event 1 encoding inside shard0: local index 1 => starts[1]:starts[2]
            s = int(shard0["starts"][1].item())
            e = int(shard0["starts"][2].item())
            c = shard0["coords"][s:e].cpu().numpy()
            f = shard0["feats"][s:e].to(torch.float32).cpu().numpy()

            self.assertEqual(c.shape, (1, 3))
            self.assertTrue((c[0] == np.array([0, 0, 0], dtype=np.int32)).all())

            # feats: [adc=log1p(1), y_norm=-1, x_norm=-1, v_norm=view-1=-1]
            self.assertAlmostEqual(float(f[0, 0]), float(np.log1p(1.0)), places=3)
            self.assertAlmostEqual(float(f[0, 1]), -1.0, places=3)
            self.assertAlmostEqual(float(f[0, 2]), -1.0, places=3)
            self.assertAlmostEqual(float(f[0, 3]), -1.0, places=3)


if __name__ == "__main__":
    unittest.main()
