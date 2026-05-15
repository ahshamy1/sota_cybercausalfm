from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_event_modeling_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy only on masked positions."""
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_labels = labels.reshape(-1)
    flat_mask = mask.reshape(-1).bool()
    if flat_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    return F.cross_entropy(flat_logits[flat_mask], flat_labels[flat_mask])


def next_event_prediction_loss(next_logits: torch.Tensor, next_labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(next_logits, next_labels)


def contrastive_alignment_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """InfoNCE over paired representations."""
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.transpose(0, 1) / temperature
    targets = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits, targets)
