import numpy as np
import torch
from typing import Optional, Tuple


def plane_to_sparse_minimal(
    plane: np.ndarray,
    view: int,
    H: int,
    W: int,
    thr: float = 0.0,
    *,
    charge_scale: Optional[float] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Convert a single (H,W) plane into sparse coordinates + minimal features.

    Coordinates (int32): [view, y, x]
    Features   (float32): [occupancy, charge, plane_id]

    Assumptions:
      - Signal is non-bipolar (charge >= 0).
      - Threshold thr is applied as plane > thr.

    Parameters
    ----------
    plane : array-like
        Plane image, any shape; will be flattened and validated against H*W.
    view : int
        Plane index (e.g., 0=U, 1=V, 2=W).
    H, W : int
        Plane dimensions.
    thr : float
        Keep pixels with value > thr.
    charge_scale : float or None
        If set, divides the transformed charge by this scale (stabilizes training).

    Returns
    -------
    coords : (N,3) int32 or None
    feats  : (N,3) float32 or None
    """
    flat = np.asarray(plane, dtype=np.float32).reshape(-1)
    if flat.size != H * W:
        raise ValueError(f"plane size {flat.size} != {H}*{W}")

    idx = np.flatnonzero(flat > thr)
    if idx.size == 0:
        return None, None

    # Map linear indices -> (y,x)
    y, x = np.divmod(idx.astype(np.int64, copy=False), W)

    coords = np.empty((idx.size, 3), dtype=np.int32)
    coords[:, 0] = np.int32(view)
    coords[:, 1] = y.astype(np.int32, copy=False)
    coords[:, 2] = x.astype(np.int32, copy=False)

    # Features: occupancy + compressed charge + plane id
    occ = np.ones((idx.size,), dtype=np.float32)

    q = flat[idx]
    q = np.log1p(q)  # non-bipolar compression
    if charge_scale is not None and charge_scale > 0:
        q = q / np.float32(charge_scale)

    # Plane identity as a simple centered scalar (U,V,W -> -1,0,1)
    plane_id = np.full((idx.size,), float(view) - 1.0, dtype=np.float32)

    feats = np.stack([occ, q.astype(np.float32, copy=False), plane_id], axis=1)
    return coords, feats


def event_to_sparse_minimal(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    H: int,
    W: int,
    thr: float = 0.0,
    *,
    charge_scale: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge U/V/W planes into one sparse event.
    Returns a non-empty placeholder if the event has no hits.
    """
    coords_list, feats_list = [], []
    for view, plane in enumerate((u, v, w)):
        c, f = plane_to_sparse_minimal(
            plane, view=view, H=H, W=W, thr=thr, charge_scale=charge_scale
        )
        if c is not None:
            coords_list.append(c)
            feats_list.append(f)

    if not coords_list:
        # Minkowski-style pipelines often dislike empty inputs.
        return (
            np.array([[0, 0, 0]], dtype=np.int32),
            np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        )

    return (
        np.concatenate(coords_list, axis=0),
        np.concatenate(feats_list, axis=0),
    )


def collate_sparse(batch):
    """
    Collate list[(coords, feats, y)] into:
      coords_batched: (sumN, 1+3) int32 with leading batch index
      feats_batched : (sumN, 3) float32
      y             : (B,) float32
    """
    coords_out, feats_out, y_out = [], [], []
    for b, (coords, feats, y) in enumerate(batch):
        c = torch.as_tensor(coords, dtype=torch.int32).contiguous()
        f = torch.as_tensor(feats, dtype=torch.float32).contiguous()
        bcol = torch.full((c.shape[0], 1), b, dtype=torch.int32)
        coords_out.append(torch.cat([bcol, c], dim=1))
        feats_out.append(f)
        y_out.append(float(y))

    return (
        torch.cat(coords_out, dim=0).contiguous(),
        torch.cat(feats_out, dim=0).contiguous(),
        torch.tensor(y_out, dtype=torch.float32),
    )
