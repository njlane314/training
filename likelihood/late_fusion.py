from __future__ import annotations

from typing import Dict, Mapping, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

BatchLike = Union[torch.Tensor, Mapping[str, torch.Tensor]]


def _forward_one(model: nn.Module, x: BatchLike) -> torch.Tensor:
    """
    Forward helper:
      - if x is a Tensor -> model(x)
      - if x is a dict-like -> model(**x)
    Must return logits shaped [B, C].
    """
    if isinstance(x, torch.Tensor):
        out = model(x)
    elif isinstance(x, Mapping):
        out = model(**x)
    else:
        raise TypeError(f"Unsupported input type: {type(x)}")

    if not isinstance(out, torch.Tensor):
        raise TypeError("Base model must return a torch.Tensor of logits [B, C].")

    if out.ndim != 2:
        raise ValueError(f"Base model output must be [B, C]. Got shape {tuple(out.shape)}")

    return out


class LateFusionClassifier(nn.Module):
    """
    Late fusion over per-modality logits.

    models: dict name -> nn.Module that outputs logits [B, C]
    fusion:
      - 'mean_logits'
      - 'weighted_logits' (learn global weights)
      - 'meta_mlp'        (stacking classifier over concatenated logits)
      - 'gated_logits'    (sample-dependent weights from logits)
    """

    def __init__(
        self,
        models: Dict[str, nn.Module],
        num_classes: int,
        fusion: str = "weighted_logits",
        meta_hidden: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not models:
            raise ValueError("models must be a non-empty dict of modality -> model")

        self.models = nn.ModuleDict(models)
        self.mod_names = list(models.keys())
        self.M = len(self.mod_names)
        self.C = int(num_classes)

        allowed = {"mean_logits", "weighted_logits", "meta_mlp", "gated_logits"}
        if fusion not in allowed:
            raise ValueError(f"fusion must be one of {sorted(allowed)}; got {fusion}")
        self.fusion = fusion

        # Global learned weights (for weighted_logits)
        if fusion == "weighted_logits":
            self.logit_weights = nn.Parameter(torch.zeros(self.M))  # softmax -> weights

        # Meta-learner (for meta_mlp): input = concatenated logits => [B, M*C]
        if fusion == "meta_mlp":
            self.meta = nn.Sequential(
                nn.Linear(self.M * self.C, meta_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(meta_hidden, self.C),
            )

        # Gating network (for gated_logits): weights per sample from logits => [B, M]
        if fusion == "gated_logits":
            self.gate = nn.Sequential(
                nn.Linear(self.M * self.C, meta_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(meta_hidden, self.M),
            )

    def forward(
        self,
        inputs: Dict[str, BatchLike],
        available_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        inputs: dict modality -> tensor or dict of tensors (for models needing multiple args)
        available_mask (optional): [B, M] boolean or {0,1} mask indicating which modalities
          are present per sample. If None, assumes all present.

        Returns: fused logits [B, C]
        """
        # Run base models
        logits_list = []
        B = None

        for m in self.mod_names:
            if m not in inputs:
                raise KeyError(f"Missing modality '{m}' in inputs. Expected keys={self.mod_names}")
            z = _forward_one(self.models[m], inputs[m])  # [B, C]

            if z.shape[1] != self.C:
                raise ValueError(
                    f"Modality '{m}' produced {z.shape[1]} classes; expected {self.C}"
                )

            if B is None:
                B = z.shape[0]
            elif z.shape[0] != B:
                raise ValueError("All modalities must have the same batch size.")

            logits_list.append(z)

        # Stack: [M, B, C]
        logits = torch.stack(logits_list, dim=0)

        # Build availability mask
        if available_mask is None:
            avail = torch.ones((B, self.M), device=logits.device, dtype=logits.dtype)
        else:
            if available_mask.shape != (B, self.M):
                raise ValueError(f"available_mask must be [B, M]=[{B},{self.M}]")
            avail = available_mask.to(device=logits.device, dtype=logits.dtype)

        # Normalize availability to avoid division by zero
        avail_sum = avail.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B,1]

        if self.fusion == "mean_logits":
            # Masked mean over modalities
            # logits: [M,B,C] -> [B,M,C]
            lbm = logits.permute(1, 0, 2)
            fused = (lbm * avail.unsqueeze(-1)).sum(dim=1) / avail_sum  # [B,C]
            return fused

        if self.fusion == "weighted_logits":
            # Global weights, but masked per sample
            w = F.softmax(self.logit_weights, dim=0)  # [M]
            # Apply per-sample availability and renormalize
            w_bm = w.unsqueeze(0).expand(B, -1) * avail  # [B,M]
            w_bm = w_bm / w_bm.sum(dim=1, keepdim=True).clamp_min(1e-8)
            fused = (logits.permute(1, 0, 2) * w_bm.unsqueeze(-1)).sum(dim=1)
            return fused

        if self.fusion == "meta_mlp":
            # Concatenate logits, but zero out missing modalities
            lbm = logits.permute(1, 0, 2) * avail.unsqueeze(-1)  # [B,M,C]
            feat = lbm.reshape(B, self.M * self.C)  # [B, M*C]
            fused = self.meta(feat)  # [B,C]
            return fused

        if self.fusion == "gated_logits":
            lbm = logits.permute(1, 0, 2)  # [B,M,C]
            feat = (lbm * avail.unsqueeze(-1)).reshape(B, self.M * self.C)
            gate_logits = self.gate(feat)  # [B,M]
            gate_w = F.softmax(gate_logits, dim=1)  # [B,M]
            # Mask + renormalize
            gate_w = gate_w * avail
            gate_w = gate_w / gate_w.sum(dim=1, keepdim=True).clamp_min(1e-8)
            fused = (lbm * gate_w.unsqueeze(-1)).sum(dim=1)  # [B,C]
            return fused

        raise RuntimeError("Unreachable: fusion mode not handled")
