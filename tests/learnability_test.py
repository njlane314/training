#!/usr/bin/env python3
"""
Learnability / sanity-check script for the sparse multi-view MinkowskiEngine model.

What it checks (from first principles):
  - If the model + data pipeline are wired correctly, it must be able to overfit
    a tiny fixed batch (training loss should collapse).
  - If it cannot overfit even a trivially learnable synthetic target derived from
    the inputs, something is wrong with inputs, masking, shapes, or gradients.

Typical usage:
  python scripts/learnability_test.py --target synth_charge --steps 300
  python scripts/learnability_test.py --target real         --steps 600
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME


# ----------------------------
# Robust local import discovery
# ----------------------------

@dataclass
class ModuleHandles:
    cfg: object
    ShardDataset: object
    collate_me_fusion: object
    BalancedBatchSampler: object
    make_backbone: object
    MultiViewSetClassifier: object


def _locate_code_root() -> Tuple[Path, Optional[str]]:
    """
    Try to locate where config.py/dataset.py/model.py/fusion.py live relative to this script.

    Returns (module_root_dir, package_prefix_or_None).
      - If files are in a flat module directory, package_prefix=None.
      - If they are in a package directory (has __init__.py), package_prefix = package name.
    """
    here = Path(__file__).resolve()
    base = here.parents[1]  # usually ".../training"
    required = ["config.py", "dataset.py", "model.py", "fusion.py"]

    candidates = [base] + [p for p in base.iterdir() if p.is_dir()]
    for cand in candidates:
        if all((cand / r).exists() for r in required):
            if (cand / "__init__.py").exists():
                return cand, cand.name
            return cand, None

    raise RuntimeError(
        f"Could not locate {required} relative to {here}. "
        f"Tried {base} and its direct subdirectories."
    )


def _import_modules() -> ModuleHandles:
    mod_root, pkg = _locate_code_root()

    # Add import paths
    if pkg is None:
        if str(mod_root) not in sys.path:
            sys.path.insert(0, str(mod_root))
        prefix = ""
    else:
        if str(mod_root.parent) not in sys.path:
            sys.path.insert(0, str(mod_root.parent))
        prefix = pkg + "."

    cfg = importlib.import_module(prefix + "config")
    dataset = importlib.import_module(prefix + "dataset")
    model = importlib.import_module(prefix + "model")
    fusion = importlib.import_module(prefix + "fusion")

    return ModuleHandles(
        cfg=cfg,
        ShardDataset=getattr(dataset, "ShardDataset"),
        collate_me_fusion=getattr(dataset, "collate_me_fusion"),
        BalancedBatchSampler=getattr(dataset, "BalancedBatchSampler"),
        make_backbone=getattr(model, "make_backbone"),
        MultiViewSetClassifier=getattr(fusion, "MultiViewSetClassifier"),
    )


# ----------------------------
# Diagnostics helpers
# ----------------------------

def _np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _tminmaxmean(name: str, x: torch.Tensor) -> str:
    x = x.detach()
    return (
        f"{name}: shape={tuple(x.shape)} dtype={x.dtype} "
        f"min={x.min().item():.4g} max={x.max().item():.4g} mean={x.float().mean().item():.4g}"
    )


def _grad_stats(model: nn.Module, eps: float = 1e-12) -> Dict[str, float]:
    n_total = 0
    n_with = 0
    n_nonzero = 0
    max_norm = 0.0
    sum_norm = 0.0
    any_nan = False

    for _, p in model.named_parameters():
        n_total += 1
        if p.grad is None:
            continue
        n_with += 1
        g = p.grad.detach()
        if not torch.isfinite(g).all():
            any_nan = True
        gn = float(g.norm().item())
        sum_norm += gn
        max_norm = max(max_norm, gn)
        if gn > eps:
            n_nonzero += 1

    return {
        "n_params": float(n_total),
        "n_with_grad": float(n_with),
        "n_nonzero_grad": float(n_nonzero),
        "mean_grad_norm_over_params_with_grad": (sum_norm / max(n_with, 1.0)),
        "max_grad_norm": max_norm,
        "any_nonfinite_grad": float(any_nan),
    }


def _param_delta_norm(before: Dict[str, torch.Tensor], model: nn.Module) -> float:
    s = 0.0
    for n, p in model.named_parameters():
        if n not in before:
            continue
        d = (p.detach() - before[n]).float()
        s += float(d.norm().item()) ** 2
    return math.sqrt(s)


def _build_inputs(
    coords_by_plane: Dict[str, torch.Tensor],
    feats_by_plane: Dict[str, torch.Tensor],
    device: torch.device,
    planes: Tuple[str, ...],
) -> Dict[str, ME.SparseTensor]:
    inputs: Dict[str, ME.SparseTensor] = {}
    for name in planes:
        coords = coords_by_plane[name]
        # Ensure coords are CPU int32
        if coords.device.type != "cpu":
            coords = coords.cpu()
        coords = coords.to(dtype=torch.int32).contiguous()

        feats = feats_by_plane[name].to(device, non_blocking=True).contiguous()
        inputs[name] = ME.SparseTensor(
            features=feats,
            coordinates=coords,
            device=device,
        )
    return inputs


def _per_event_reduce(
    coords: torch.Tensor,
    feats: torch.Tensor,
    B: int,
    which: str,
) -> torch.Tensor:
    """
    Reduce per-event either:
      - which='nnz': count occupancy>0 sites
      - which='charge': sum logq over occupancy>0 sites
    coords: [N,3] (batch,y,x)
    feats:  [N,2] (occ, logq)
    """
    b = coords[:, 0].to(torch.int64)
    occ = feats[:, 0] > 0
    out = torch.zeros((B,), dtype=torch.float32)
    if which == "nnz":
        out.index_add_(0, b[occ], torch.ones_like(b[occ], dtype=torch.float32))
        return out
    if which == "charge":
        out.index_add_(0, b[occ], feats[occ, 1].to(torch.float32))
        return out
    raise ValueError(which)


def _make_target(
    target: str,
    y_real: torch.Tensor,
    coords_by_plane: Dict[str, torch.Tensor],
    feats_by_plane: Dict[str, torch.Tensor],
    planes: Tuple[str, ...],
) -> torch.Tensor:
    """
    Returns y in {0,1} float32, shape [B].
    """
    if target == "real":
        return y_real.clone()

    B = int(y_real.shape[0])

    # Compute a simple scalar from inputs: total charge (sum of logq) across all planes
    total_charge = torch.zeros((B,), dtype=torch.float32)
    total_nnz = torch.zeros((B,), dtype=torch.float32)
    for name in planes:
        total_charge += _per_event_reduce(coords_by_plane[name], feats_by_plane[name], B, which="charge")
        total_nnz += _per_event_reduce(coords_by_plane[name], feats_by_plane[name], B, which="nnz")

    if target == "synth_charge":
        # Median threshold -> roughly balanced even if the distribution is skewed.
        med = torch.median(total_charge)
        y = (total_charge > med).to(torch.float32)
        # Edge case: all equal (rare but possible). Fall back to nnz.
        if y.min().item() == y.max().item():
            med2 = torch.median(total_nnz)
            y = (total_nnz > med2).to(torch.float32)
        return y

    if target == "shuffled":
        perm = torch.randperm(B)
        return y_real[perm].clone()

    raise ValueError(f"unknown --target {target!r}")


# ----------------------------
# Main routine
# ----------------------------

def main():
    mh = _import_modules()
    cfg = mh.cfg

    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-dir", default=getattr(cfg, "SHARDS_DIR", "shards"))
    ap.add_argument("--backbone", default=getattr(cfg, "BACKBONE", "small"))
    ap.add_argument("--embed-dim", type=int, default=int(getattr(cfg, "EMBED_DIM", 256)))
    ap.add_argument("--batch", type=int, default=int(getattr(cfg, "BATCH_SIZE", 16)))
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=int(getattr(cfg, "SEED", 123)))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])
    ap.add_argument("--target", default="synth_charge", choices=["real", "synth_charge", "shuffled"])
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--no-amp", action="store_true", help="Force disable AMP even on CUDA (for debugging).")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device(args.device)

    shards_dir = Path(args.shards_dir)
    idx_path = shards_dir / "index.pt"
    if not idx_path.exists():
        raise FileNotFoundError(f"Missing {idx_path}. Did you run your shard writer?")

    meta = torch.load(idx_path, map_location="cpu")
    labels = _np(meta["labels"]).astype(np.uint8).reshape(-1)
    weights = _np(meta["weights"]).astype(np.float32).reshape(-1) if "weights" in meta else None
    n = int(meta.get("n_events", labels.shape[0]))

    nnz_all = None
    if "nnz" in meta and meta["nnz"] is not None:
        nnz_all = _np(meta["nnz"]).astype(np.int64).reshape(-1)
        if nnz_all.shape[0] != n:
            print(f"[warn] meta['nnz'] len={nnz_all.shape[0]} != n_events={n}; ignoring nnz filter.")
            nnz_all = None

    bad_events = _np(meta.get("bad_events", np.array([], dtype=np.int64))).reshape(-1)

    sig_frac = float(labels[:n].mean())
    print(f"[meta] shards_dir={shards_dir}")
    print(f"[meta] n_events={n}  signal_fraction={sig_frac:.4f}  (sig={labels.sum()} bkg={(labels==0).sum()})")
    if weights is not None:
        w = weights[:n]
        print(f"[meta] weights: min={w.min():.4g} max={w.max():.4g} mean={w.mean():.4g}  <=0: {(w<=0).sum()}")
    if nnz_all is not None:
        z = nnz_all[:n]
        print(f"[meta] nnz: min={z.min()} max={z.max()} mean={z.mean():.2f} zeros={(z==0).sum()} ({(z==0).mean()*100:.3f}%)")
    print(f"[meta] bad_events={bad_events.size}")

    # Filter to good events if nnz provided
    if nnz_all is not None:
        good_mask = nnz_all[:n] > 0
        idx_good = np.flatnonzero(good_mask)
    else:
        idx_good = np.arange(n, dtype=np.int64)

    sig = idx_good[labels[idx_good] == 1]
    bkg = idx_good[labels[idx_good] == 0]
    if sig.size == 0 or bkg.size == 0:
        raise RuntimeError("Need both signal and background after filtering to run the tests.")

    if args.batch % 2 != 0:
        raise ValueError("--batch must be even (to pick a balanced fixed batch).")

    h = args.batch // 2
    pick_sig = rng.choice(sig, size=h, replace=(sig.size < h))
    pick_bkg = rng.choice(bkg, size=h, replace=(bkg.size < h))
    fixed_idx = np.concatenate([pick_sig, pick_bkg]).astype(np.int64)
    rng.shuffle(fixed_idx)

    # Load exactly one fixed batch (single worker for reproducibility)
    ds = mh.ShardDataset(str(shards_dir), fixed_idx, cache_size=2)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
        collate_fn=mh.collate_me_fusion,
        pin_memory=(device.type == "cuda"),
    )

    planes = ("u", "v", "w")
    (coords_by_plane, feats_by_plane, y_real, available_mask) = next(iter(dl))
    B = int(y_real.shape[0])

    # Basic batch sanity
    print(f"[batch] B={B}  y_real: mean={y_real.mean().item():.3f}  (sig={(y_real>0.5).sum().item()} bkg={(y_real<0.5).sum().item()})")
    print(f"[batch] available_mask: per-plane sums={available_mask.sum(dim=0).tolist()}  per-event min={available_mask.sum(dim=1).min().item()} max={available_mask.sum(dim=1).max().item()}")

    real_hits_total = 0
    for name in planes:
        c = coords_by_plane[name]
        f = feats_by_plane[name]
        # Count real hits (occ>0), not dummy sites
        real_hits = int((f[:, 0] > 0).sum().item())
        real_hits_total += real_hits
        print(f"[batch] plane={name}  coords={tuple(c.shape)} feats={tuple(f.shape)}  real_hits(occ>0)={real_hits}  logq min/max={(f[f[:,0]>0,1].min().item() if real_hits else 0.0):.4g}/{(f[f[:,0]>0,1].max().item() if real_hits else 0.0):.4g}")

        # Per-event nnz summary
        nnz_evt = _per_event_reduce(c, f, B, which="nnz")
        print(f"        per-event nnz: min={nnz_evt.min().item():.0f} mean={nnz_evt.mean().item():.1f} max={nnz_evt.max().item():.0f}")

    if real_hits_total == 0:
        print("[FAIL] This batch contains ZERO real hits across all planes (only dummy sites).")
        print("       That would make logits stick near 0 and kill gradients through ReLUs.")
        print("       Check: THRESH, branch names, shard generation, index.pt nnz, and collate masking.")
        sys.exit(2)

    # Build model
    backbone = mh.make_backbone(args.backbone, in_ch=2, embed_dim=args.embed_dim).to(device)
    model = mh.MultiViewSetClassifier(backbone=backbone, embed_dim=args.embed_dim, plane_names=planes).to(device)

    # Capture pooled embedding variance via hook
    captured: Dict[str, torch.Tensor] = {}

    def _pool_hook(mod, inp, out):
        captured["pooled"] = out.detach()

    if hasattr(model, "pool"):
        model.pool.register_forward_hook(_pool_hook)

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    # Make chosen target
    y = _make_target(args.target, y_real, coords_by_plane, feats_by_plane, planes).to(torch.float32)
    y = y.to(device, non_blocking=True)
    available_mask_dev = available_mask.to(device, non_blocking=True)

    # One-step gradient sanity
    model.train()
    inputs = _build_inputs(coords_by_plane, feats_by_plane, device, planes)

    logits = model(inputs, available_mask=available_mask_dev).squeeze(1)
    if logits.shape != y.shape:
        raise RuntimeError(f"logits shape {tuple(logits.shape)} != y shape {tuple(y.shape)}")

    loss = loss_fn(logits, y)
    with torch.no_grad():
        p = torch.sigmoid(logits)
        acc = ((p > 0.5) == (y > 0.5)).float().mean().item()
        print(_tminmaxmean("[init] logits", logits))
        print(_tminmaxmean("[init] prob", p))
        if "pooled" in captured:
            pooled = captured["pooled"]
            print(_tminmaxmean("[init] pooled", pooled))
            # variance across batch is the key "depends on input?" signal
            print(f"[init] pooled std over batch={pooled.float().std(dim=0).mean().item():.4g}  (mean std per-dim)")
        print(f"[init] loss={loss.item():.6f}  acc={acc:.3f}  target={args.target}")

    opt.zero_grad(set_to_none=True)
    loss.backward()

    gstat = _grad_stats(model)
    print(
        "[grad] "
        f"params={int(gstat['n_params'])} with_grad={int(gstat['n_with_grad'])} nonzero_grad={int(gstat['n_nonzero_grad'])} "
        f"mean_grad_norm={gstat['mean_grad_norm_over_params_with_grad']:.4g} max_grad_norm={gstat['max_grad_norm']:.4g} "
        f"any_nonfinite={bool(gstat['any_nonfinite_grad'])}"
    )

    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt.step()
    delta = _param_delta_norm(before, model)
    print(f"[step] param_delta_L2={delta:.6g}")

    if int(gstat["n_nonzero_grad"]) == 0 or delta == 0.0:
        print("[FAIL] No effective learning signal: gradients are all ~0 and/or parameters did not change.")
        print("       Most common causes in this setup:")
        print("         - inputs are effectively all zeros (or only dummy sites survive collate/masking)")
        print("         - available_mask accidentally masks everything (all views unavailable)")
        print("         - logits are constant AND network is 'dead' via ReLU/normalization on zero inputs")
        sys.exit(3)

    # Overfit loop on the fixed batch
    print(f"[overfit] training on ONE fixed batch for {args.steps} steps (lr={args.lr}, wd={args.weight_decay}, target={args.target})")
    for step in range(1, args.steps + 1):
        model.train()
        inputs = _build_inputs(coords_by_plane, feats_by_plane, device, planes)
        logits = model(inputs, available_mask=available_mask_dev).squeeze(1)
        loss = loss_fn(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            with torch.no_grad():
                p = torch.sigmoid(logits)
                acc = ((p > 0.5) == (y > 0.5)).float().mean().item()
                lstd = logits.std().item()
            print(f"[overfit] step {step:5d}  loss {loss.item():.6f}  acc {acc:.3f}  logits_std {lstd:.4g}")

    with torch.no_grad():
        model.eval()
        inputs = _build_inputs(coords_by_plane, feats_by_plane, device, planes)
        logits = model(inputs, available_mask=available_mask_dev).squeeze(1)
        loss = loss_fn(logits, y).item()
        p = torch.sigmoid(logits)
        acc = ((p > 0.5) == (y > 0.5)).float().mean().item()

    print(f"[result] final loss={loss:.6f}  final acc={acc:.3f}  target={args.target}")

    # Interpret result heuristically
    if acc > 0.95 and loss < 0.2:
        print("[PASS] The model can overfit a fixed batch => gradients + data flow are OK (the model is learnable).")
    else:
        print("[WARN] Overfit did not reach high accuracy on a fixed batch.")
        print("       Interpretation:")
        print("         - If --target synth_charge also fails: pipeline/gradient bug is likely.")
        print("         - If synth_charge passes but real fails: labels/problem may be hard or mis-specified.")
        print("         - If real passes here but full training is stuck: optimizer/lr/schedule/sampler issue in train.py.")


if __name__ == "__main__":
    main()
