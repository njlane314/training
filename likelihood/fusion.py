from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

import MinkowskiEngine as ME


class ViewAttentionPool(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(inplace=True),
            nn.Linear(d, 1),
        )

    def forward(self, Z: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        s = self.score(Z).squeeze(-1)
        if mask is not None:
            m = mask.to(dtype=torch.bool, device=Z.device)
            s = s.masked_fill(~m, -1e9)
        else:
            m = None
        w = torch.softmax(s, dim=1)
        if m is not None:
            w = w * m.to(dtype=w.dtype)
            w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (w.unsqueeze(-1) * Z).sum(dim=1)


class MultiViewSetClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, embed_dim: int, plane_names: Tuple[str, ...] = ("u", "v", "w")):
        super().__init__()
        self.backbone = backbone
        self.plane_names = tuple(plane_names)
        self.num_views = len(self.plane_names)
        self.embed_dim = int(embed_dim)
        self.plane_emb = nn.Embedding(self.num_views, self.embed_dim)
        nn.init.zeros_(self.plane_emb.weight)
        self.pool = ViewAttentionPool(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, 1)

    def forward(
        self,
        inputs: Dict[str, ME.SparseTensor],
        available_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        Z = []
        for pid, name in enumerate(self.plane_names):
            x = inputs[name]
            z = self.backbone(x)
            z = z + self.plane_emb.weight[pid].unsqueeze(0)
            Z.append(z)
        Z = torch.stack(Z, dim=1)
        pooled = self.pool(Z, mask=available_mask)
        return self.head(pooled)
