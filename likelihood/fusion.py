from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn


def _call(model: nn.Module, x: Any) -> torch.Tensor:
    if isinstance(x, Mapping):
        out = model(**x)
    else:
        out = model(x)
    if not isinstance(out, torch.Tensor):
        raise TypeError("Base model must return a torch.Tensor.")
    if out.ndim == 1:
        out = out.unsqueeze(1)
    if out.ndim != 2:
        raise ValueError(f"Expected logits [B,C]. Got {tuple(out.shape)}")
    return out


class LateFusionClassifier(nn.Module):
    """
    Product-of-experts (PoE) fusion in logit space.

    If each per-plane model is trained with balanced class prior (p(sig)=p(bkg)=0.5),
    its output logit is an estimator of the per-plane log-likelihood ratio (LLR).
    Under conditional independence, the fused LLR is the sum of available logits.

    We include a tiny calibration head: fused = exp(log_scale) * sum_logits + bias.
    """

    def __init__(
        self,
        models: Dict[str, nn.Module],
        num_classes: int = 1,
    ):
        super().__init__()
        if not models:
            raise ValueError("models must be a non-empty dict of modality -> model")

        self.models = nn.ModuleDict(models)
        self.mod_names = list(models.keys())
        self.M = len(self.mod_names)
        self.C = int(num_classes)

        self.log_scale = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(self.C))

    def forward(
        self,
        inputs: Dict[str, Any],
        available_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        inputs: dict modality -> model input (e.g. ME.SparseTensor)
        available_mask: [B, M] float/bool mask (1 if modality present for sample)

        Returns: fused logits [B, C].
        """
        logits_list = []
        B = None

        for m in self.mod_names:
            if m not in inputs:
                raise KeyError(f"Missing modality '{m}' in inputs. Expected keys={self.mod_names}")
            z = _call(self.models[m], inputs[m])  # [B,C]

            if z.shape[1] != self.C:
                raise ValueError(
                    f"Modality '{m}' produced {z.shape[1]} classes; expected {self.C}"
                )

            if B is None:
                B = z.shape[0]
            elif z.shape[0] != B:
                raise ValueError("All modalities must have the same batch size.")

            logits_list.append(z)

        # [B,M,C]
        logits = torch.stack(logits_list, dim=1)

        if available_mask is None:
            avail = torch.ones((B, self.M), device=logits.device, dtype=logits.dtype)
        else:
            if available_mask.shape != (B, self.M):
                raise ValueError(f"available_mask must be [B, M]=[{B},{self.M}]")
            avail = available_mask.to(device=logits.device, dtype=logits.dtype)

        sum_logits = (logits * avail.unsqueeze(-1)).sum(dim=1)  # [B,C]
        fused = torch.exp(self.log_scale) * sum_logits + self.bias
        return fused
