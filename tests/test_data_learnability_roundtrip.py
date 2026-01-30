import os
import sys
import tempfile
import subprocess
import unittest

import numpy as np
import torch
import torch.nn as nn

import uproot

from likelihood.data import event_to_sparse, ShardDataset


def auc_pairwise(y_true, scores):
    """
    AUC = P(score_pos > score_neg) + 0.5 P(tie).
    O(n_pos*n_neg), fine for small tests.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if pos.size == 0 or neg.size == 0:
        raise ValueError("need both classes for AUC")

    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return (gt + 0.5 * eq) / (pos.size * neg.size)


def make_synthetic_root(root_path, *, n_events=32, H=16, W=16, hits_per_view=8, seed=0):
    """
    Build a tiny ROOT file with a *guaranteed* learnable difference:
      - same nnz structure for signal/background (so nnz can't solve it)
      - but adc amplitudes differ strongly (signal >> background)
    """
    rng = np.random.default_rng(seed)

    u = np.zeros((n_events, H * W), dtype=np.float32)
    v = np.zeros_like(u)
    w = np.zeros_like(u)

    labels = np.zeros(n_events, dtype=np.uint8)
    weights = np.ones(n_events, dtype=np.float32)

    for i in range(n_events):
        y = 1 if (i % 2 == 0) else 0  # balanced
        labels[i] = y

        # enforce identical hit-count pattern across classes
        # but different amplitudes => learnable from adc, not from nnz
        amp = 5.0 if y == 1 else 1.0

        for plane in (u, v, w):
            idx = rng.choice(H * W, size=hits_per_view, replace=False)
            plane[i, idx] = amp

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

    return labels, weights


def run_write_shards_subprocess(*, root_file, shards_out, H, W, thr=0.0, signlog=False, shard_events=8, chunk_events=8):
    """
    Run likelihood.data.write_shards_from_root() in a subprocess so env-based config is clean.
    """
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    env = os.environ.copy()
    env["PYTHONPATH"] = proj_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.update(
        {
            "ROOT_FILE": root_file,
            "TREE": "events",
            "SHARDS_DIR": shards_out,
            "SHARDS_OUT": shards_out,
            "H": str(int(H)),
            "W": str(int(W)),
            "THRESH": str(float(thr)),
            "ADC_SIGNLOG": "1" if signlog else "0",
            "SHARD_EVENTS": str(int(shard_events)),
            "CHUNK_EVENTS": str(int(chunk_events)),
            "UPROOT_DECOMP_WORKERS": "0",
        }
    )

    code = "from likelihood.data import write_shards_from_root; write_shards_from_root()"
    r = subprocess.run(
        [sys.executable, "-c", code],
        cwd=proj_root,
        env=env,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"write_shards_from_root failed\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")


def shard_summaries(ds: ShardDataset):
    """
    Compute simple per-event summary features from ShardDataset output.
    """
    y = np.zeros(len(ds), dtype=np.int64)
    nnz = np.zeros(len(ds), dtype=np.int64)
    sum_adc = np.zeros(len(ds), dtype=np.float64)
    max_adc = np.zeros(len(ds), dtype=np.float64)

    for i in range(len(ds)):
        c, f, yi = ds[i]
        y[i] = int(yi)
        nnz[i] = int(f.shape[0])
        sum_adc[i] = float(f[:, 0].sum().item())
        max_adc[i] = float(f[:, 0].max().item())

    return y, nnz, sum_adc, max_adc


class TestDataDifferenceAndLearnability(unittest.TestCase):
    def test_signal_background_difference_roundtrip_and_learnability(self):
        """
        End-to-end data test (no MinkowskiEngine required):

        (1) Build a synthetic ROOT where signal/background differ in amplitude but not nnz.
        (2) Run ROOT->shards writer.
        (3) Verify labels preserved in index.pt.
        (4) Verify per-event coords/features match ROOT-derived sparse conversion (within float16 tolerance).
        (5) Verify signal/background are separable after reading shards (AUC on sum_adc is high).
        (6) Verify a trivial baseline (logistic regression on summary features) learns (loss decreases).
        """
        H = W = 16
        n_events = 32
        hits_per_view = 8
        thr = 0.0
        signlog = False

        with tempfile.TemporaryDirectory() as tmp:
            root_path = os.path.join(tmp, "events.root")
            shards_out = os.path.join(tmp, "shards")
            os.makedirs(shards_out, exist_ok=True)

            labels_true, weights_true = make_synthetic_root(
                root_path,
                n_events=n_events,
                H=H,
                W=W,
                hits_per_view=hits_per_view,
                seed=0,
            )

            run_write_shards_subprocess(
                root_file=root_path,
                shards_out=shards_out,
                H=H,
                W=W,
                thr=thr,
                signlog=signlog,
                shard_events=8,   # force multiple shards
                chunk_events=8,
            )

            idx = torch.load(os.path.join(shards_out, "index.pt"), map_location="cpu")
            self.assertEqual(int(idx["n_events"]), n_events)
            self.assertEqual(int(idx["H"]), H)
            self.assertEqual(int(idx["W"]), W)

            # (3) labels/weights preserved
            self.assertTrue(torch.equal(idx["labels"].to(torch.uint8), torch.from_numpy(labels_true)))
            self.assertTrue(torch.allclose(idx["weights"].to(torch.float32), torch.from_numpy(weights_true)))

            # Read shards via dataset
            ds = ShardDataset(shards_out, np.arange(n_events, dtype=np.int64), cache_size=2)

            # (5) difference + separability after reading
            y, nnz, sum_adc, max_adc = shard_summaries(ds)

            # labels preserved through ShardDataset
            self.assertTrue(np.array_equal(y.astype(np.uint8), labels_true))

            # nnz should be identical across classes by construction (3 views * hits_per_view)
            expected_nnz = 3 * hits_per_view
            self.assertTrue(np.all(nnz == expected_nnz), msg=f"nnz not constant: min={nnz.min()} max={nnz.max()}")

            # sanity: for this construction, sum_adc(signal) >> sum_adc(bkg)
            mu_sig = float(sum_adc[y == 1].mean())
            mu_bkg = float(sum_adc[y == 0].mean())
            self.assertGreater(mu_sig - mu_bkg, 10.0, msg=f"unexpectedly small separation: mu_sig={mu_sig:.3f} mu_bkg={mu_bkg:.3f}")

            # AUC using a simple statistic
            auc_sum = auc_pairwise(y, sum_adc)
            self.assertGreater(auc_sum, 0.95, msg=f"AUC(sum_adc) too low: {auc_sum:.3f}")

            # AUC using nnz should be ~0.5 since nnz is constant
            auc_nnz = auc_pairwise(y, nnz.astype(np.float64))
            self.assertLess(abs(auc_nnz - 0.5), 1e-6, msg=f"AUC(nnz) should be 0.5 but got {auc_nnz:.6f}")

            # (4) Round-trip check: ROOT -> event_to_sparse matches dataset per-event (coords exact, feats ~close)
            with uproot.open(root_path) as f:
                t = f["events"]
                a = t.arrays(
                    ["detector_image_u", "detector_image_v", "detector_image_w"],
                    entry_start=0,
                    entry_stop=n_events,
                    library="np",
                )

            uu = a["detector_image_u"]
            vv = a["detector_image_v"]
            ww = a["detector_image_w"]

            check = [0, 1, 7, 8, 15]  # cross shard boundaries
            for i in check:
                c_root, f_root = event_to_sparse(uu[i], vv[i], ww[i], H=H, W=W, thr=thr, signlog=signlog)
                c_ds, f_ds, _ = ds[i]

                c_root = np.asarray(c_root, dtype=np.int32)
                f_root = np.asarray(f_root, dtype=np.float32)
                c_ds_np = c_ds.cpu().numpy().astype(np.int32)
                f_ds_np = f_ds.cpu().numpy().astype(np.float32)

                self.assertTrue(
                    np.array_equal(c_ds_np, c_root),
                    msg=f"coords mismatch at event {i}",
                )
                # float16 round-trip => allow small absolute error
                self.assertTrue(
                    np.allclose(f_ds_np, f_root, atol=1e-2, rtol=0.0),
                    msg=f"feats mismatch at event {i}: max_abs={np.max(np.abs(f_ds_np - f_root))}",
                )

            # (6) Learnability test: logistic regression on summary features learns (loss decreases)
            X = np.stack([sum_adc, max_adc], axis=1).astype(np.float32)
            Y = y.astype(np.float32).reshape(-1, 1)

            Xt = torch.tensor(X, dtype=torch.float32)
            Yt = torch.tensor(Y, dtype=torch.float32)

            # standardize for stable optimization (not required but removes brittle dependence on scales)
            Xt = (Xt - Xt.mean(dim=0, keepdim=True)) / Xt.std(dim=0, keepdim=True).clamp_min(1e-6)

            torch.manual_seed(0)
            lin = nn.Linear(Xt.shape[1], 1)
            opt = torch.optim.Adam(lin.parameters(), lr=0.1)
            loss_fn = nn.BCEWithLogitsLoss()

            with torch.no_grad():
                loss0 = float(loss_fn(lin(Xt), Yt).item())

            for _ in range(100):
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(lin(Xt), Yt)
                loss.backward()
                opt.step()

            with torch.no_grad():
                loss1 = float(loss_fn(lin(Xt), Yt).item())

            self.assertLess(loss1, loss0 - 0.2, msg=f"baseline did not learn: loss0={loss0:.4f} loss1={loss1:.4f}")


if __name__ == "__main__":
    unittest.main()
