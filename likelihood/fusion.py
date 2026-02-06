from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

import MinkowskiEngine as ME


class ViewAttentionPool(nn.Module):
    """
    Permutation-invariant (set) pooling over a small number of views.
    Z: [B, V, D] -> pooled: [B, D]
    """

    def __init__(self, d: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d, d),
            # In-place ReLU can interfere with some attribution methods that rely on
            # hooks / multiple backward passes.
            nn.ReLU(inplace=False),
            nn.Linear(d, 1),
        )

    def forward(self, Z: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # scores: [B,V]
        s = self.score(Z).squeeze(-1)

        if mask is not None:
            m = mask.to(dtype=torch.bool, device=Z.device)
            # If an event has *no* available views (all False), masking would
            # zero-out attention weights and kill gradients into Z/backbone.
            # Fall back to treating all views as available for those rows.
            none = ~m.any(dim=1, keepdim=True)  # [B,1]
            if none.any():
                m = m.clone()
                m[none.expand_as(m)] = True
            s = s.masked_fill(~m, -1e9)
        else:
            m = None

        w = torch.softmax(s, dim=1)  # [B,V]
        if m is not None:
            w = w * m.to(dtype=w.dtype)
            w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-6)

        return (w.unsqueeze(-1) * Z).sum(dim=1)  # [B,D]


class MultiViewSetClassifier(nn.Module):
    """
    Shared 2D sparse backbone per plane + learnable, order-invariant pooling across planes.

    - backbone(x: ME.SparseTensor) -> embedding [B, D]
    - add a (learned) plane embedding per view
    - attention-pool across the set of view embeddings
    - linear head -> logits [B,1]
    """

    def __init__(self, backbone: nn.Module, embed_dim: int, plane_names: Tuple[str, ...] = ("u", "v", "w")):
        super().__init__()
        self.backbone = backbone
        self.plane_names = tuple(plane_names)
        self.num_views = len(self.plane_names)
        self.embed_dim = int(embed_dim)

        self.plane_emb = nn.Embedding(self.num_views, self.embed_dim)
        nn.init.zeros_(self.plane_emb.weight)  # start symmetric; learn differences if useful

        self.pool = ViewAttentionPool(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, 1)

    def forward(
        self,
        inputs: Dict[str, ME.SparseTensor],
        available_mask: Optional[torch.Tensor] = None,  # [B,V] from collate_me_fusion
    ) -> torch.Tensor:
        Z = []
        for pid, name in enumerate(self.plane_names):
            x = inputs[name]
            z = self.backbone(x)  # [B,D]
            z = z + self.plane_emb.weight[pid].unsqueeze(0)  # [B,D]
            Z.append(z)

        Z = torch.stack(Z, dim=1)  # [B,V,D]
        pooled = self.pool(Z, mask=available_mask)  # [B,D]
        return self.head(pooled)  # [B,1]
