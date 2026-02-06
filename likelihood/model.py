from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

import MinkowskiEngine as ME


def _replace_feature(x: ME.SparseTensor, new_F: torch.Tensor) -> ME.SparseTensor:
    """
    Compatibility wrapper for SparseTensor feature replacement.

    Newer MinkowskiEngine versions provide `SparseTensor.replace_feature`.
    Older versions require re-instantiating a SparseTensor while *sharing*
    the same coordinate map + manager (coords_key/coords_man or
    coordinate_map_key/coordinate_manager).
    """
    if hasattr(x, "replace_feature"):
        return x.replace_feature(new_F)

    # Collect whatever this ME build exposes on the tensor.
    cmk = getattr(x, "coordinate_map_key", None)
    cm = getattr(x, "coordinate_manager", None)
    ck = getattr(x, "coords_key", None)
    # Older builds commonly use `coords_man`; some use `coords_manager`.
    cman = getattr(x, "coords_man", None)
    if cman is None:
        cman = getattr(x, "coords_manager", None)

    # Try "new" ctor keywords first, then legacy ones.
    if cmk is not None and cm is not None:
        try:
            return ME.SparseTensor(
                features=new_F,
                coordinate_map_key=cmk,
                coordinate_manager=cm,
                device=new_F.device,
            )
        except TypeError:
            pass

    if ck is not None and cman is not None:
        try:
            return ME.SparseTensor(
                features=new_F,
                coords_key=ck,
                coords_manager=cman,
                device=new_F.device,
            )
        except TypeError:
            pass

    # As a last resort, rebuild from explicit coordinates (may create a new
    # coordinate manager; this is less ideal but avoids hard failure).
    return ME.SparseTensor(
        features=new_F,
        coordinates=x.C,
        device=new_F.device,
    )


def _subm_conv(
    in_ch: int,
    out_ch: int,
    *,
    kernel_size: int,
    dimension: int,
    bias: bool = False,
) -> nn.Module:
    """
    Compatibility wrapper for submanifold sparse convolution.

    Some MinkowskiEngine builds don't expose `MinkowskiSubmanifoldConvolution`.
    The equivalent behavior is `MinkowskiConvolution(..., expand_coordinates=False)`
    (which is the default in many versions, but we try to pass it explicitly).
    """
    if hasattr(ME, "MinkowskiSubmanifoldConvolution"):
        return ME.MinkowskiSubmanifoldConvolution(
            int(in_ch), int(out_ch), kernel_size=kernel_size, dimension=dimension, bias=bias
        )
    try:
        return ME.MinkowskiConvolution(
            int(in_ch),
            int(out_ch),
            kernel_size=kernel_size,
            stride=1,
            dimension=dimension,
            bias=bias,
            expand_coordinates=False,
        )
    except TypeError:
        # Older ME builds may not accept `expand_coordinates`.
        return ME.MinkowskiConvolution(
            int(in_ch), int(out_ch), kernel_size=kernel_size, stride=1, dimension=dimension, bias=bias
        )


class SparseLayerNorm(nn.Module):
    """
    Batch-size/occupancy agnostic normalization:
    LayerNorm over channels per sparse site (point).
    """

    def __init__(self, c: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(int(c), eps=eps)

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        return _replace_feature(x, self.ln(x.F))


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv1 = _subm_conv(cin, cout, kernel_size=3, dimension=2, bias=False)
        self.n1 = SparseLayerNorm(cout)
        self.conv2 = _subm_conv(cout, cout, kernel_size=3, dimension=2, bias=False)
        self.n2 = SparseLayerNorm(cout)
        # NOTE:
        # In-place activations are a common source of surprises for gradient-based
        # attribution methods (multiple backward passes / hooks / captum-style tools).
        # Keeping this non-inplace makes attribution maps stable.
        self.act = ME.MinkowskiReLU(inplace=False)
        self.proj = None
        if cin != cout:
            self.proj = nn.Sequential(
                ME.MinkowskiLinear(cin, cout, bias=False),
                SparseLayerNorm(cout),
            )

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        identity = x if self.proj is None else self.proj(x)
        out = self.act(self.n1(self.conv1(x)))
        out = self.n2(self.conv2(out))
        return self.act(out + identity)


class Down(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(
            ME.MinkowskiConvolution(cin, cout, kernel_size=2, stride=2, dimension=2, bias=False),
            SparseLayerNorm(cout),
            ME.MinkowskiReLU(inplace=False),
        )


class SparseResNet2D(nn.Module):
    """
    Minimal 2D sparse residual encoder -> global pooled embedding.
    backbone(x) returns [B, D] (not logits), for late/set fusion.
    """

    def __init__(self, in_ch: int, base: int, blocks: Tuple[int, ...], embed_dim: int):
        super().__init__()
        self.stem = nn.Sequential(
            _subm_conv(in_ch, base, kernel_size=3, dimension=2, bias=False),
            SparseLayerNorm(base),
            ME.MinkowskiReLU(inplace=False),
        )
        self.blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = int(base)
        for li, nb in enumerate(blocks):
            self.blocks.append(nn.Sequential(*[ResBlock(ch, ch) for _ in range(int(nb))]))
            if li < len(blocks) - 1:
                self.downs.append(Down(ch, ch * 2))
                ch *= 2
        self.pool = ME.MinkowskiGlobalMaxPooling()
        self.proj = nn.Linear(ch, int(embed_dim))

    def forward(self, x: ME.SparseTensor) -> torch.Tensor:
        x = self.stem(x)
        for li, blk in enumerate(self.blocks):
            x = blk(x)
            if li < len(self.downs):
                x = self.downs[li](x)
        x = self.pool(x)
        return self.proj(x.F)  # [B,D]


_PRESETS: Dict[str, Dict] = {
    # (base channels, blocks per level); 512x512 -> 3 or 4 downs is typical
    "tiny": {"base": 16, "blocks": (1, 1, 1)},
    "small": {"base": 32, "blocks": (1, 1, 1, 1)},
    "base": {"base": 32, "blocks": (2, 2, 2, 2)},
    "wide": {"base": 48, "blocks": (2, 2, 2, 2)},
}


def make_backbone(name: str, in_ch: int, embed_dim: int) -> nn.Module:
    """
    Factory for quick backbone sweeps:
      BACKBONE={tiny,small,base,wide}
    """
    if name not in _PRESETS:
        raise ValueError(f"unknown BACKBONE={name!r}; choose from {sorted(_PRESETS)}")
    p = _PRESETS[name]
    return SparseResNet2D(in_ch=int(in_ch), base=int(p["base"]), blocks=tuple(p["blocks"]), embed_dim=int(embed_dim))
