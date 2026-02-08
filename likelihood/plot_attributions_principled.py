#!/usr/bin/env python3
"""
plot_attributions_principled.py

More principled attribution for MultiViewSetClassifier:

  1) Integrated Gradients (IG)
     - Choose a baseline x' (explicit, user-controlled).
     - Integrate gradients along the straight-line path:
           x(α) = x' + α (x - x'),   α ∈ [0,1]
     - Per-feature IG:
           IG = (x - x') * ∫_0^1 ∂objective/∂x(α) dα
     - This is gradient-based but less brittle than single-point gradients.

     IMPORTANT (sparse setting): IG here keeps the *sparse coordinate set fixed* and
     uses the event's available_mask as-is. So IG answers:
         "Given these hit coordinates (and available views), how do feature values
          along the path from baseline -> input change the objective?"
     It does NOT remove/add coordinates during the path.

  2) Occlusion / perturbation tests (delta-logit / delta-objective)
     - For each hit (or patch), ablate it and re-run the model.
     - Measure Δ (logit or objective).
     - This is more "causal" but can be expensive.

Outputs:
  A 2x3 grid (u,v,w):
    row 0: input intensity (sum over input channels)
    row 1: attribution map (IG or occlusion Δ)

Example usage:
  # Charge-only IG (baseline: logq=0, keep occ as-is)
  python likelihood/plot_attributions_principled.py \
    --ckpt checkpoints/checkpoint_step0010000.pt \
    --method ig --baseline zero_logq --channel logq --ig-steps 64 \
    --target true --random-signal

  # Occlusion test: zero out logq at each hit (subset) and plot |Δobjective|
  python likelihood/plot_attributions_principled.py \
    --ckpt checkpoints/checkpoint_step0010000.pt \
    --method occlusion --occlusion-unit site --occlude-mode zero --channel logq \
    --max-occlusions 1500 --metric objective --map-mode abs --target true --random-signal

  # Occlusion test: drop hits in 16x16 patches and plot signed Δobjective
  python likelihood/plot_attributions_principled.py \
    --ckpt checkpoints/checkpoint_step0010000.pt \
    --method occlusion --occlusion-unit patch --patch 16 --occlude-mode drop \
    --metric objective --map-mode signed --target true --random-signal
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME

# Robust imports whether run as a script or as a module inside a package.
try:
    from . import config as cfg  # type: ignore
    from .dataset import ShardDataset, collate_me_fusion  # type: ignore
    from .fusion import MultiViewSetClassifier  # type: ignore
    from .model import make_backbone  # type: ignore
except Exception:
    import config as cfg  # type: ignore
    from dataset import ShardDataset, collate_me_fusion  # type: ignore
    from fusion import MultiViewSetClassifier  # type: ignore
    from model import make_backbone  # type: ignore


PLANES: Tuple[str, ...] = ("u", "v", "w")


def _load_meta(shards_dir: str) -> dict:
    return torch.load(f"{shards_dir}/index.pt", map_location="cpu")


def _compute_splits_from_meta(
    meta: dict,
    *,
    seed: int,
    val_fraction: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n = int(meta["n_events"])
    rng = np.random.default_rng(int(seed))
    labels_all = np.asarray(meta["labels"], dtype=np.uint8).reshape(-1)
    if labels_all.shape[0] != n:
        raise ValueError(f"index.pt labels len={labels_all.shape[0]} != n_events={n}")

    nnz_all: Optional[np.ndarray] = None
    if "nnz" in meta and meta["nnz"] is not None:
        if isinstance(meta["nnz"], torch.Tensor):
            nnz_all = meta["nnz"].to(dtype=torch.int64).cpu().numpy().reshape(-1)
        else:
            nnz_all = np.asarray(meta["nnz"], dtype=np.int64).reshape(-1)
        if nnz_all.shape[0] != n:
            raise ValueError(f"index.pt nnz len={nnz_all.shape[0]} != n_events={n}")

    if nnz_all is not None:
        good = nnz_all > 0
        idx_all = np.flatnonzero(good)
        if idx_all.size == 0:
            raise ValueError("All events have nnz==0 after sparsification.")
        perm = rng.permutation(idx_all.size)
        idx_perm = idx_all[perm]
        n_val = int(float(val_fraction) * idx_perm.size)
        val_idx = idx_perm[:n_val]
        train_idx = idx_perm[n_val:]
        return train_idx.astype(np.int64), val_idx.astype(np.int64)

    perm = rng.permutation(n)
    n_val = int(float(val_fraction) * n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


def _sort_event_indices_for_io(meta: dict, event_idx: np.ndarray) -> np.ndarray:
    shard_events = int(meta.get("shard_events", getattr(cfg, "SHARD_EVENTS", 2048)))
    shard_id = (event_idx // shard_events).astype(np.int64, copy=False)
    local_id = (event_idx - shard_id * shard_events).astype(np.int64, copy=False)
    key = np.lexsort((local_id, shard_id))
    return event_idx[key].astype(np.int64, copy=False)


def _build_model_from_cfg(device: torch.device) -> nn.Module:
    backbone = make_backbone(cfg.BACKBONE, in_ch=2, embed_dim=cfg.EMBED_DIM).to(device)
    model = MultiViewSetClassifier(backbone=backbone, embed_dim=cfg.EMBED_DIM, plane_names=PLANES).to(device)
    return model


def _sparse_to_dense_2d(
    coords_byx: np.ndarray,
    values: np.ndarray,
    *,
    out_shape: Tuple[int, int],
    batch_index: int = 0,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    coords_byx: [N,3] (batch,y,x) (matches process.py + collate_me_fusion)
    values:     [N]
    out_shape:  (H,W)
    """
    H, W = int(out_shape[0]), int(out_shape[1])
    if coords_byx.size == 0:
        img = np.zeros((H, W), dtype=np.float32)
        extent = (-0.5, float(W) - 0.5, -0.5, float(H) - 0.5)
        return img, extent

    c = np.asarray(coords_byx)
    if c.ndim != 2 or c.shape[1] != 3:
        raise ValueError(f"coords must have shape [N,3], got {c.shape}")

    b = c[:, 0].astype(np.int64, copy=False)
    sel = (b == int(batch_index))
    img = np.zeros((H, W), dtype=np.float32)
    if not np.any(sel):
        extent = (-0.5, float(W) - 0.5, -0.5, float(H) - 0.5)
        return img, extent

    y = c[sel, 1].astype(np.int64, copy=False)
    x = c[sel, 2].astype(np.int64, copy=False)
    v = np.asarray(values, dtype=np.float32).reshape(-1)[sel]

    inb = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    if np.any(inb):
        np.add.at(img, (y[inb], x[inb]), v[inb])

    extent = (-0.5, float(W) - 0.5, -0.5, float(H) - 0.5)
    return img, extent


def _nonzero_percentiles(img: np.ndarray, *, q_lo: float, q_hi: float) -> Tuple[Optional[float], Optional[float]]:
    nz = np.asarray(img, dtype=np.float32).reshape(-1)
    nz = nz[np.isfinite(nz) & (nz != 0)]
    if nz.size == 0:
        return None, None
    lo = float(np.percentile(nz, float(q_lo)))
    hi = float(np.percentile(nz, float(q_hi)))
    return lo, hi


def _make_baseline(feats: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Baselines are defined on the same sparse coordinates (no coordinate deletion).

    - zero      : all channels -> 0
    - zero_logq : keep occ channel as-is, set logq channel -> 0 (recommended for charge-only IG)
    """
    if mode == "zero":
        return torch.zeros_like(feats)
    if mode == "zero_logq":
        if feats.ndim != 2 or feats.shape[1] < 2:
            raise ValueError(f"zero_logq baseline requires feats [N,>=2], got {tuple(feats.shape)}")
        base = feats.clone()
        base[:, 1] = 0.0
        return base
    raise ValueError(f"unknown baseline mode: {mode!r}")


def _choose_target_class(target: str, *, logit: float, y_true: int) -> int:
    if target == "pred":
        return 1 if logit > 0.0 else 0
    if target == "true":
        return int(y_true)
    if target == "1":
        return 1
    return 0


def _objective_from_logit(logit: torch.Tensor, target_class: int) -> torch.Tensor:
    return logit if int(target_class) == 1 else -logit


def _reduce_channels(
    tensor_nc: torch.Tensor,
    *,
    channel: str,
    signed: bool,
) -> torch.Tensor:
    """
    tensor_nc: [N,C]
    channel: both|occ|logq
    signed:
      - True : sum over selected channels (keeps sign)
      - False: sum over abs(selected channels)
    """
    if tensor_nc.ndim != 2:
        raise ValueError(f"expected [N,C], got {tuple(tensor_nc.shape)}")
    if channel == "both":
        t = tensor_nc
    elif channel == "occ":
        t = tensor_nc[:, 0:1]
    else:  # logq
        if tensor_nc.shape[1] < 2:
            raise ValueError(f"logq channel requested but C={int(tensor_nc.shape[1])} < 2")
        t = tensor_nc[:, 1:2]
    return t.sum(dim=1) if signed else t.abs().sum(dim=1)


def _ensure_nonempty_plane(
    coords: torch.Tensor,
    feats: torch.Tensor,
    *,
    n_feat: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MinkowskiEngine can be fragile with empty SparseTensors in some builds.
    Guarantee at least one dummy site with zero features.
    """
    if int(coords.shape[0]) > 0:
        return coords, feats
    dummy_c = torch.zeros((1, 3), dtype=torch.int32)
    dummy_f = torch.zeros((1, int(n_feat)), dtype=torch.float32)
    return dummy_c, dummy_f


def _available_mask_from_feats(feats_by_plane_cpu: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Recompute [1,3] available_mask from features (CPU tensors), matching collate_me_fusion logic:
      available if any occ > 0 in that plane.
    """
    m = torch.zeros((1, len(PLANES)), dtype=torch.float32)
    for pi, name in enumerate(PLANES):
        f = feats_by_plane_cpu[name]
        if f.ndim == 2 and f.shape[1] >= 1 and (f[:, 0] > 0).any().item():
            m[0, pi] = 1.0
    return m


def _pick_event_idx(args: argparse.Namespace, meta: dict) -> int:
    """
    Matches the selection logic in plot_attributions.py (event-idx, val-rank, random signal/background).
    """
    if args.event_idx is not None:
        return int(args.event_idx)

    train_idx, val_idx = _compute_splits_from_meta(meta, seed=int(cfg.SEED), val_fraction=float(cfg.VAL_FRACTION))

    n_events = int(meta["n_events"])
    labels_all = np.asarray(meta["labels"], dtype=np.uint8).reshape(-1)
    if labels_all.shape[0] != n_events:
        raise ValueError(f"index.pt labels len={labels_all.shape[0]} != n_events={n_events}")

    # "all" pool respects nnz>0 filtering if available
    if "nnz" in meta and meta["nnz"] is not None:
        if isinstance(meta["nnz"], torch.Tensor):
            nnz_all = meta["nnz"].to(dtype=torch.int64).cpu().numpy().reshape(-1)
        else:
            nnz_all = np.asarray(meta["nnz"], dtype=np.int64).reshape(-1)
        good_idx = np.flatnonzero(nnz_all > 0).astype(np.int64, copy=False)
    else:
        good_idx = np.arange(n_events, dtype=np.int64)

    do_random = bool(args.random or args.random_signal or args.random_background)
    if do_random:
        if args.random_from == "val":
            pool = val_idx
        elif args.random_from == "train":
            pool = train_idx
        else:
            pool = good_idx

        want_label: Optional[int] = None
        if args.random_signal:
            want_label = 1
        elif args.random_background:
            want_label = 0

        if want_label is not None:
            pool = pool[labels_all[pool] == np.uint8(want_label)]

        if pool.size == 0:
            raise ValueError(
                f"No events available for random selection: random_from={args.random_from} "
                f"label={want_label if want_label is not None else 'any'}"
            )

        rng = np.random.default_rng(None if args.rng_seed is None else int(args.rng_seed))
        event_idx = int(rng.choice(pool))
        print(
            f"[pick] random from {args.random_from}: event_idx={event_idx} "
            f"(y={int(labels_all[event_idx])}) seed={args.rng_seed}"
        )
        return event_idx

    # Deterministic val-rank (IO-sorted)
    val_idx = _sort_event_indices_for_io(meta, val_idx)
    vr = int(args.val_rank)
    if vr < 0 or vr >= int(val_idx.size):
        raise ValueError(f"--val-rank out of range: {vr} (val size={val_idx.size})")
    return int(val_idx[vr])


def _forward_logit_single(
    model: nn.Module,
    coords_by_plane: Dict[str, torch.Tensor],
    feats_by_plane_dev: Dict[str, torch.Tensor],
    available_mask: torch.Tensor,
    device: torch.device,
    *,
    require_grad: bool,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, ME.SparseTensor]]:
    """
    Build ME inputs and run forward for batch_size=1.

    Returns:
      logit: scalar tensor
      feats_leaf: dict of [N,C] tensors (only meaningful if require_grad=True)
      inputs: dict of ME.SparseTensor
    """
    inputs: Dict[str, ME.SparseTensor] = {}
    feats_leaf: Dict[str, torch.Tensor] = {}
    for name in PLANES:
        f = feats_by_plane_dev[name]
        if require_grad:
            f = f.detach().requires_grad_(True)
        inputs[name] = ME.SparseTensor(features=f, coordinates=coords_by_plane[name], device=device)
        feats_leaf[name] = f

    with torch.cuda.amp.autocast(enabled=False):
        logits = model(inputs, available_mask=available_mask.to(device, non_blocking=True)).squeeze(1)
        if logits.numel() != 1:
            raise RuntimeError(f"Expected batch_size=1, got logits shape {tuple(logits.shape)}")
        return logits[0], feats_leaf, inputs


def _integrated_gradients(
    model: nn.Module,
    coords_by_plane: Dict[str, torch.Tensor],
    feats_by_plane_cpu: Dict[str, torch.Tensor],
    available_mask: torch.Tensor,
    device: torch.device,
    *,
    target_class: int,
    baseline_mode: str,
    steps: int,
) -> Tuple[Dict[str, torch.Tensor], float, float]:
    """
    Returns:
      ig_by_plane: dict -> [N,C] IG tensor (signed per feature)
      obj_input: objective at input (float)
      obj_base : objective at baseline (float)
    """
    if steps <= 0:
        raise ValueError("--ig-steps must be > 0")

    # Move input feats to device (no grad).
    feats_in: Dict[str, torch.Tensor] = {k: v.to(device, non_blocking=True) for k, v in feats_by_plane_cpu.items()}
    feats_base: Dict[str, torch.Tensor] = {k: _make_baseline(v, baseline_mode).to(device) for k, v in feats_in.items()}
    delta: Dict[str, torch.Tensor] = {k: (feats_in[k] - feats_base[k]) for k in PLANES}

    # Objective at input and baseline (no-grad)
    with torch.no_grad():
        logit_in, _, _ = _forward_logit_single(
            model, coords_by_plane, feats_in, available_mask, device, require_grad=False
        )
        logit_b, _, _ = _forward_logit_single(
            model, coords_by_plane, feats_base, available_mask, device, require_grad=False
        )
        obj_in = float(_objective_from_logit(logit_in, target_class).detach().cpu())
        obj_b = float(_objective_from_logit(logit_b, target_class).detach().cpu())

    # Trapezoidal rule over α in [0,1]:
    #   integral ≈ (1/steps) * (0.5*g(0) + Σ_{k=1..steps-1} g(k/steps) + 0.5*g(1))
    sum_grads: Dict[str, torch.Tensor] = {k: torch.zeros_like(feats_in[k]) for k in PLANES}

    for si in range(0, steps + 1):
        alpha = float(si) / float(steps)
        weight = 0.5 if (si == 0 or si == steps) else 1.0

        feats_a: Dict[str, torch.Tensor] = {}
        for name in PLANES:
            fa = feats_base[name] + alpha * delta[name]
            feats_a[name] = fa

        logit_a, feats_leaf, _ = _forward_logit_single(
            model, coords_by_plane, feats_a, available_mask, device, require_grad=True
        )
        obj_a = _objective_from_logit(logit_a, target_class)

        grads = torch.autograd.grad(
            outputs=obj_a,
            inputs=[feats_leaf[name] for name in PLANES],
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        for name, g in zip(PLANES, grads):
            if g is None:
                continue
            sum_grads[name] += float(weight) * g.detach()

    ig: Dict[str, torch.Tensor] = {}
    for name in PLANES:
        avg_grad = sum_grads[name] / float(steps)
        ig[name] = delta[name] * avg_grad  # [N,C], signed

    return ig, obj_in, obj_b


@torch.no_grad()
def _occlusion_deltas(
    model: nn.Module,
    coords_by_plane: Dict[str, torch.Tensor],
    feats_by_plane_cpu: Dict[str, torch.Tensor],
    available_mask_cpu: torch.Tensor,
    device: torch.device,
    *,
    target_class: int,
    metric: str,            # "objective" or "logit"
    occlude_mode: str,      # "zero" or "drop"
    occlusion_unit: str,    # "site" or "patch"
    patch: int,
    channel: str,           # both|occ|logq
    baseline_mode: str,     # used only for occlude_mode="zero"
    max_occlusions: Optional[int],
    rng_seed: Optional[int],
) -> Tuple[Dict[str, np.ndarray], float]:
    """
    Compute perturbation deltas per plane.

    Returns:
      maps_by_plane: dict -> dense [H,W] float32 map (signed deltas)
      base_value: objective or logit at the original input (float)
    """
    # Base forward
    feats_in_dev: Dict[str, torch.Tensor] = {k: v.to(device, non_blocking=True) for k, v in feats_by_plane_cpu.items()}
    logit0, _, _ = _forward_logit_single(
        model, coords_by_plane, feats_in_dev, available_mask_cpu.to(device), device, require_grad=False
    )
    logit0_f = float(logit0.detach().cpu())
    obj0_f = float(_objective_from_logit(logit0, target_class).detach().cpu())
    base_value = obj0_f if metric == "objective" else logit0_f

    # Baseline replacement (for "zero" mode)
    baseline_cpu: Dict[str, torch.Tensor] = {
        k: _make_baseline(v, baseline_mode).detach().cpu() for k, v in feats_by_plane_cpu.items()
    }

    rng = np.random.default_rng(None if rng_seed is None else int(rng_seed))

    maps: Dict[str, np.ndarray] = {}

    for pi, name in enumerate(PLANES):
        is_avail = bool(available_mask_cpu[0, pi].item())
        # If a plane is missing, just return an all-zeros map.
        # (Still plot is handled by caller similarly.)
        if not is_avail:
            maps[name] = None  # type: ignore[assignment]
            continue

        coords0 = coords_by_plane[name].detach().cpu()
        feats0 = feats_by_plane_cpu[name].detach().cpu()
        C = int(feats0.shape[1])

        # Eligible = "real hits" (occ > 0). Avoid dummy sites.
        elig = (feats0[:, 0] > 0).numpy()
        idx_all = np.flatnonzero(elig).astype(np.int64, copy=False)

        if idx_all.size == 0:
            maps[name] = None  # type: ignore[assignment]
            continue

        # Allocate dense map at patch or pixel resolution later; fill zeros by default.
        # Here we build a per-plane dense [H,W] image at the end in the caller (needs H/W).
        # We'll store per-site deltas for "site" unit, and per-patch deltas in a patch image for "patch" unit.

        # Determine which indices/patches to test
        if occlusion_unit == "site":
            idx_test = idx_all
            if max_occlusions is not None and idx_test.size > int(max_occlusions):
                idx_test = rng.choice(idx_test, size=int(max_occlusions), replace=False)
                idx_test = np.sort(idx_test)

            # Per-site delta values aligned with coords0 rows (un-tested rows remain 0).
            delta_per_site = np.zeros((int(coords0.shape[0]),), dtype=np.float32)

            for k, idx in enumerate(idx_test.tolist()):
                idx = int(idx)

                # Build modified coords/feats for THIS plane; other planes unchanged.
                if occlude_mode == "drop":
                    keep = np.ones((int(coords0.shape[0]),), dtype=bool)
                    keep[idx] = False
                    coords_mod = coords0[keep]
                    feats_mod = feats0[keep]
                else:
                    coords_mod = coords0
                    feats_mod = feats0.clone()
                    b = baseline_cpu[name]
                    if channel == "both":
                        feats_mod[idx, :] = b[idx, :]
                    elif channel == "occ":
                        feats_mod[idx, 0] = b[idx, 0]
                    else:  # logq
                        feats_mod[idx, 1] = b[idx, 1]

                coords_mod, feats_mod = _ensure_nonempty_plane(coords_mod, feats_mod, n_feat=C)

                feats_by_plane_mod_cpu = {
                    pname: (feats_mod if pname == name else feats_by_plane_cpu[pname])
                    for pname in PLANES
                }
                coords_by_plane_mod = {
                    pname: (coords_mod if pname == name else coords_by_plane[pname])
                    for pname in PLANES
                }
                am = _available_mask_from_feats(feats_by_plane_mod_cpu)

                feats_mod_dev = {pname: feats_by_plane_mod_cpu[pname].to(device, non_blocking=True) for pname in PLANES}
                logit1, _, _ = _forward_logit_single(
                    model, coords_by_plane_mod, feats_mod_dev, am.to(device), device, require_grad=False
                )
                logit1_f = float(logit1.detach().cpu())
                obj1_f = float(_objective_from_logit(logit1, target_class).detach().cpu())

                val1 = obj1_f if metric == "objective" else logit1_f
                delta = float(base_value - val1)
                delta_per_site[idx] = np.float32(delta)

                if (k + 1) % 200 == 0 or (k + 1) == len(idx_test):
                    print(f"[occlusion:{name}] {k+1}/{len(idx_test)}")

            maps[name] = delta_per_site  # still per-site; caller rasterizes

        else:
            # Patch occlusion: ablate all eligible hits within each patch, measure delta, fill a dense patch image.
            if int(patch) <= 0:
                raise ValueError("--patch must be > 0 for patch occlusion")

            y = coords0[:, 1].numpy().astype(np.int64, copy=False)
            x = coords0[:, 2].numpy().astype(np.int64, copy=False)

            # We'll fill a dense patch image in the caller; here store per-patch dict (pid -> delta).
            # But simplest is to store a per-site array with patch-constant values (eligible sites only),
            # then rasterize to dense with _sparse_to_dense_2d.
            pid = (y // int(patch)) * (10_000_000) + (x // int(patch))  # unique id without needing W
            pid_elig = pid[elig]
            uniq = np.unique(pid_elig)

            if max_occlusions is not None and uniq.size > int(max_occlusions):
                uniq = rng.choice(uniq, size=int(max_occlusions), replace=False)
                uniq = np.sort(uniq)

            delta_per_site = np.zeros((int(coords0.shape[0]),), dtype=np.float32)

            for k, u in enumerate(uniq.tolist()):
                u = int(u)
                in_patch = (pid == u) & elig

                if occlude_mode == "drop":
                    keep = ~in_patch
                    coords_mod = coords0[keep]
                    feats_mod = feats0[keep]
                else:
                    coords_mod = coords0
                    feats_mod = feats0.clone()
                    b = baseline_cpu[name]
                    if channel == "both":
                        feats_mod[in_patch, :] = b[in_patch, :]
                    elif channel == "occ":
                        feats_mod[in_patch, 0] = b[in_patch, 0]
                    else:
                        feats_mod[in_patch, 1] = b[in_patch, 1]

                coords_mod, feats_mod = _ensure_nonempty_plane(coords_mod, feats_mod, n_feat=C)

                feats_by_plane_mod_cpu = {
                    pname: (feats_mod if pname == name else feats_by_plane_cpu[pname])
                    for pname in PLANES
                }
                coords_by_plane_mod = {
                    pname: (coords_mod if pname == name else coords_by_plane[pname])
                    for pname in PLANES
                }
                am = _available_mask_from_feats(feats_by_plane_mod_cpu)

                feats_mod_dev = {pname: feats_by_plane_mod_cpu[pname].to(device, non_blocking=True) for pname in PLANES}
                logit1, _, _ = _forward_logit_single(
                    model, coords_by_plane_mod, feats_mod_dev, am.to(device), device, require_grad=False
                )
                logit1_f = float(logit1.detach().cpu())
                obj1_f = float(_objective_from_logit(logit1, target_class).detach().cpu())
                val1 = obj1_f if metric == "objective" else logit1_f
                delta = float(base_value - val1)

                # Assign the patch delta to eligible sites in that patch (for rasterization).
                delta_per_site[in_patch] = np.float32(delta)

                if (k + 1) % 100 == 0 or (k + 1) == len(uniq):
                    print(f"[occlusion:{name}] {k+1}/{len(uniq)} patches")

            maps[name] = delta_per_site  # per-site but patch-constant on eligible hits

    return maps, float(base_value)


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", type=str, required=True, help="Checkpoint path (ckpt_step*.pt).")
    ap.add_argument("--out-dir", type=str, default="attrib")
    ap.add_argument("--out-name", type=str, default="attributions_principled.png")

    ap.add_argument(
        "--method",
        type=str,
        required=True,
        choices=("ig", "occlusion"),
        help="Attribution method: integrated gradients (ig) or perturbation-based deltas (occlusion).",
    )

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--event-idx", type=int, default=None, help="Global event index to visualize.")
    g.add_argument("--val-rank", type=int, default=0, help="If no --event-idx: deterministic val split rank.")
    g.add_argument("--random", action="store_true", help="Pick a random event from --random-from.")
    g.add_argument("--random-signal", action="store_true", help="Pick a random signal event (y=1).")
    g.add_argument("--random-background", action="store_true", help="Pick a random background event (y=0).")

    ap.add_argument("--random-from", type=str, default="val", choices=("val", "train", "all"))
    ap.add_argument("--rng-seed", type=int, default=None)

    ap.add_argument(
        "--target",
        type=str,
        default="pred",
        choices=("pred", "true", "1", "0"),
        help="Objective target class. Uses objective=+logit for class 1, objective=-logit for class 0.",
    )

    ap.add_argument(
        "--channel",
        type=str,
        default="both",
        choices=("both", "occ", "logq"),
        help="Which input channel(s) to reduce over for plotting (and which channel(s) to ablate for occlusion).",
    )

    ap.add_argument(
        "--map-mode",
        type=str,
        default="abs",
        choices=("abs", "signed"),
        help="How to visualize attributions: abs (LogNorm on |value|) or signed (diverging around 0).",
    )
    ap.add_argument("--qlo", type=float, default=5.0, help="Lower percentile for color scaling (nonzero pixels).")
    ap.add_argument("--qhi", type=float, default=99.5, help="Upper percentile for color scaling (nonzero pixels).")

    # IG options
    ap.add_argument("--ig-steps", type=int, default=64, help="Number of IG integration steps (trapezoidal).")
    ap.add_argument(
        "--baseline",
        type=str,
        default="zero_logq",
        choices=("zero", "zero_logq"),
        help="IG baseline definition on fixed coordinates. zero_logq keeps occ and sets logq=0 (recommended).",
    )

    # Occlusion options
    ap.add_argument(
        "--metric",
        type=str,
        default="objective",
        choices=("objective", "logit"),
        help="What delta to report for occlusion: objective (signed for target) or raw logit.",
    )
    ap.add_argument(
        "--occlusion-unit",
        type=str,
        default="patch",
        choices=("site", "patch"),
        help="Occlude individual hits (site) or groups via patches (patch).",
    )
    ap.add_argument("--patch", type=int, default=16, help="Patch size for --occlusion-unit patch.")
    ap.add_argument(
        "--occlude-mode",
        type=str,
        default="zero",
        choices=("zero", "drop"),
        help=(
            "Occlusion type: "
            "zero sets selected channel(s) to baseline values but keeps the coordinate; "
            "drop removes the hit(s) (drops coordinates) for that plane."
        ),
    )
    ap.add_argument(
        "--max-occlusions",
        type=int,
        default=None,
        help="Cap how many occlusions to run per plane (random subset). Applies to both site and patch.",
    )

    ap.add_argument("--num-workers", type=int, default=0)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / str(args.out_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta = _load_meta(cfg.SHARDS_DIR)
    H = int(meta.get("H", getattr(cfg, "H", 512)))
    W = int(meta.get("W", getattr(cfg, "W", 512)))

    event_idx = _pick_event_idx(args, meta)

    # Load model + checkpoint.
    model = _build_model_from_cfg(device)
    ck = torch.load(args.ckpt, map_location="cpu")
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(state, strict=True)
    model.eval()

    # Load event.
    ds = ShardDataset(cfg.SHARDS_DIR, np.asarray([event_idx], dtype=np.int64), cache_size=2)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate_me_fusion,
        pin_memory=True,
        persistent_workers=False,
    )
    batch = next(iter(dl))
    coords_by_plane, feats_by_plane_cpu, y, available_mask = batch
    y0 = int(y[0].item())

    # Forward once on the real input to choose a fixed target_class (must be fixed for IG and occlusion comparisons).
    feats_in_dev = {k: v.to(device, non_blocking=True) for k, v in feats_by_plane_cpu.items()}
    logit0_t, _, _ = _forward_logit_single(
        model, coords_by_plane, feats_in_dev, available_mask.to(device), device, require_grad=False
    )
    logit0 = float(logit0_t.detach().cpu())
    target_class = _choose_target_class(args.target, logit=logit0, y_true=y0)

    # Compute attribution values
    if args.method == "ig":
        ig_by_plane, obj_in, obj_b = _integrated_gradients(
            model,
            coords_by_plane,
            feats_by_plane_cpu,
            available_mask.to(device),
            device,
            target_class=target_class,
            baseline_mode=str(args.baseline),
            steps=int(args.ig_steps),
        )

        # Completeness sanity check (signed sum of IG over all features)
        ig_sum = 0.0
        for name in PLANES:
            ig_sum += float(ig_by_plane[name].sum().detach().cpu())
        print(
            f"[ig] obj(input)={obj_in:+.6g}  obj(base)={obj_b:+.6g}  "
            f"delta={obj_in-obj_b:+.6g}  sum(IG)={ig_sum:+.6g}  diff={(obj_in-obj_b-ig_sum):+.6g}"
        )

        # Per-plane per-site scalar values for plotting
        attr_vals: Dict[str, np.ndarray] = {}
        for name in PLANES:
            signed = (args.map_mode == "signed")
            v = _reduce_channels(ig_by_plane[name], channel=str(args.channel), signed=signed)
            attr_vals[name] = v.detach().cpu().numpy().astype(np.float32, copy=False)

        base_value = float(_objective_from_logit(logit0_t, target_class).detach().cpu())
        title_suffix = f"IG baseline={args.baseline} steps={int(args.ig_steps)}"

    else:
        maps_by_plane, base_value = _occlusion_deltas(
            model,
            coords_by_plane,
            feats_by_plane_cpu,
            available_mask.detach().cpu(),
            device,
            target_class=target_class,
            metric=str(args.metric),
            occlude_mode=str(args.occlude_mode),
            occlusion_unit=str(args.occlusion_unit),
            patch=int(args.patch),
            channel=str(args.channel),
            baseline_mode=str(args.baseline),
            max_occlusions=(None if args.max_occlusions is None else int(args.max_occlusions)),
            rng_seed=args.rng_seed,
        )
        attr_vals = maps_by_plane  # per-site deltas aligned with coords rows
        title_suffix = (
            f"occlusion unit={args.occlusion_unit} mode={args.occlude_mode} "
            f"metric={args.metric} channel={args.channel}"
        )

    # Plot
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, TwoSlopeNorm

    fig, axs = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)

    for ci, name in enumerate(PLANES):
        ax_in = axs[0, ci]
        ax_at = axs[1, ci]

        is_avail = bool(available_mask[0, ci].item())
        if not is_avail:
            ax_in.axis("off")
            ax_at.axis("off")
            ax_in.text(0.5, 0.5, f"{name}: missing", ha="center", va="center", transform=ax_in.transAxes)
            continue

        coords = coords_by_plane[name].detach().cpu().numpy()
        feats = feats_by_plane_cpu[name].detach()

        inp_val = feats.sum(dim=1).detach().cpu().numpy()
        img_in, extent = _sparse_to_dense_2d(coords, inp_val, out_shape=(H, W))

        ax_in.imshow(img_in, origin="lower", extent=extent, aspect="auto", interpolation="nearest")
        ax_in.set_title(f"{name} input")
        ax_in.set_xlabel("x")
        ax_in.set_ylabel("y")

        # Attribution map
        v = attr_vals[name]
        if v is None:
            ax_at.axis("off")
            continue

        img_at, extent = _sparse_to_dense_2d(coords, v, out_shape=(H, W))

        if args.map_mode == "abs":
            show = np.abs(img_at.astype(np.float32, copy=False))
            vlo, vhi = _nonzero_percentiles(show, q_lo=float(args.qlo), q_hi=float(args.qhi))
            if vlo is None or vhi is None:
                ax_at.imshow(show, origin="lower", extent=extent, aspect="auto", interpolation="nearest")
            else:
                vmin = max(float(vlo), 1e-12)
                vmax = max(float(vhi), vmin * 1.001)
                masked = np.ma.masked_less_equal(show, 0.0)
                ax_at.imshow(
                    masked,
                    origin="lower",
                    extent=extent,
                    aspect="auto",
                    interpolation="nearest",
                    norm=LogNorm(vmin=vmin, vmax=vmax),
                )
        else:
            show = img_at.astype(np.float32, copy=False)
            abs_show = np.abs(show)
            vlo, vhi = _nonzero_percentiles(abs_show, q_lo=float(args.qlo), q_hi=float(args.qhi))
            if vhi is None:
                ax_at.imshow(show, origin="lower", extent=extent, aspect="auto", interpolation="nearest")
            else:
                vlim = max(float(vhi), 1e-12)
                norm = TwoSlopeNorm(vcenter=0.0, vmin=-vlim, vmax=+vlim)
                ax_at.imshow(show, origin="lower", extent=extent, aspect="auto", interpolation="nearest", norm=norm)

        ax_at.set_title(f"{name} attribution ({args.method}, {args.map_mode})")
        ax_at.set_xlabel("x")
        ax_at.set_ylabel("y")

    obj0 = float(_objective_from_logit(logit0_t, target_class).detach().cpu())
    fig.suptitle(
        f"event_idx={event_idx}  y={y0}  target={target_class}  "
        f"logit={logit0:+.3f}  obj={obj0:+.3f}  base({args.metric})={base_value:+.3f}\n{title_suffix}"
    )
    fig.savefig(str(out_path), dpi=160)
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
