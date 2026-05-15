from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List

import torch
import torch.nn.functional as F

from .model import CyberCausalFM
from .objectives import (
    contrastive_alignment_loss,
    masked_event_modeling_loss,
    next_event_prediction_loss,
)


@dataclass
class ScalingPoint:
    event_dim: int
    n_layers: int
    n_heads: int
    train_steps: int
    data_fraction: float
    final_loss: float
    in_domain_acc: float
    zero_day_proxy_acc: float
    params: int


def _build_batch(batch_size: int, seq_len: int, vocab_size: int, num_classes: int) -> tuple[torch.Tensor, ...]:
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = tokens.clone()
    mask = torch.rand(batch_size, seq_len) < 0.15
    masked_tokens = tokens.clone()
    masked_tokens[mask] = 0

    graph_x = torch.randn(batch_size, 12, 32)
    adj = torch.rand(batch_size, 12, 12)
    adj = (adj + adj.transpose(1, 2)) / 2.0

    next_labels = torch.randint(0, vocab_size, (batch_size,))
    class_labels = torch.randint(0, num_classes, (batch_size,))
    return masked_tokens, labels, mask, graph_x, adj, next_labels, class_labels


def _proxy_eval(model: CyberCausalFM, vocab_size: int, num_classes: int, zero_day: bool = False) -> float:
    model.eval()
    with torch.no_grad():
        tokens = torch.randint(0, vocab_size, (128, 24))
        if zero_day:
            tokens = (tokens + (vocab_size // 2)) % vocab_size
        graph_x = torch.randn(128, 12, 32)
        adj = torch.rand(128, 12, 12)
        adj = (adj + adj.transpose(1, 2)) / 2.0
        labels = torch.randint(0, num_classes, (128,))
        logits = model(tokens, graph_x, adj)["class_logits"]
        pred = logits.argmax(dim=-1)
        return float((pred == labels).float().mean().item())


def run_scaling_law_experiment(seed: int = 13) -> List[ScalingPoint]:
    torch.manual_seed(seed)

    configs = [
        # (event_dim, n_layers, n_heads, data_fraction, train_steps)
        (64, 1, 2, 0.3, 20),
        (128, 2, 4, 0.6, 30),
        (192, 3, 6, 1.0, 40),
    ]

    points: List[ScalingPoint] = []
    vocab_size = 128
    num_classes = 6

    for event_dim, n_layers, n_heads, data_fraction, train_steps in configs:
        model = CyberCausalFM(
            event_vocab_size=vocab_size,
            event_dim=event_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            num_classes=num_classes,
        )
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

        final_loss = 0.0
        for _ in range(train_steps):
            batch_size = max(8, int(32 * data_fraction))
            batch = _build_batch(batch_size, seq_len=24, vocab_size=vocab_size, num_classes=num_classes)
            masked_tokens, labels, mask, graph_x, adj, next_labels, class_labels = batch

            out = model(masked_tokens, graph_x, adj)
            l_mcm = masked_event_modeling_loss(out["token_logits"], labels, mask)
            l_nap = next_event_prediction_loss(out["next_event_logits"], next_labels)
            z1 = out["fusion"]
            z2 = out["fusion"] + 0.01 * torch.randn_like(out["fusion"])
            l_align = contrastive_alignment_loss(z1, z2)
            l_cls = F.cross_entropy(out["class_logits"], class_labels)

            loss = l_mcm + l_nap + 0.2 * l_align + l_cls
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            final_loss = float(loss.detach().cpu().item())

        in_acc = _proxy_eval(model, vocab_size=vocab_size, num_classes=num_classes, zero_day=False)
        zd_acc = _proxy_eval(model, vocab_size=vocab_size, num_classes=num_classes, zero_day=True)
        params = sum(p.numel() for p in model.parameters())

        points.append(
            ScalingPoint(
                event_dim=event_dim,
                n_layers=n_layers,
                n_heads=n_heads,
                train_steps=train_steps,
                data_fraction=data_fraction,
                final_loss=final_loss,
                in_domain_acc=in_acc,
                zero_day_proxy_acc=zd_acc,
                params=params,
            )
        )

    return points


def scaling_points_to_rows(points: List[ScalingPoint]) -> List[dict]:
    return [asdict(p) for p in points]
