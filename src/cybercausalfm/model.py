from __future__ import annotations

import torch
import torch.nn as nn


class SimpleGraphEncoder(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, hid_dim),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B, N, F], adj: [B, N, N]
        agg = torch.bmm(adj, x)
        return self.proj(agg).mean(dim=1)


class CyberCausalFM(nn.Module):
    """SOTA-style compact model: temporal + graph + uncertainty heads."""

    def __init__(
        self,
        event_vocab_size: int = 128,
        event_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        graph_feat_dim: int = 32,
        fusion_dim: int = 192,
        num_classes: int = 6,
    ) -> None:
        super().__init__()
        self.event_emb = nn.Embedding(event_vocab_size, event_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=event_dim,
            nhead=n_heads,
            dim_feedforward=event_dim * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.temporal = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.graph = SimpleGraphEncoder(graph_feat_dim, event_dim)

        self.fusion = nn.Sequential(
            nn.Linear(event_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.token_head = nn.Linear(event_dim, event_vocab_size)
        self.next_event_head = nn.Linear(event_dim, event_vocab_size)
        self.classifier = nn.Linear(fusion_dim, num_classes)

        # Mean and log-variance for uncertainty estimation.
        self.uncertainty_head = nn.Linear(fusion_dim, 2)

    def forward(
        self,
        event_tokens: torch.Tensor,
        graph_x: torch.Tensor,
        graph_adj: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        emb = self.event_emb(event_tokens)
        h = self.temporal(emb)

        token_logits = self.token_head(h)
        seq_repr = h.mean(dim=1)
        next_event_logits = self.next_event_head(seq_repr)

        g_repr = self.graph(graph_x, graph_adj)
        fused = self.fusion(torch.cat([seq_repr, g_repr], dim=-1))

        class_logits = self.classifier(fused)
        unc = self.uncertainty_head(fused)
        mu = torch.sigmoid(unc[:, :1])
        log_var = unc[:, 1:2]

        return {
            "token_logits": token_logits,
            "next_event_logits": next_event_logits,
            "class_logits": class_logits,
            "fusion": fused,
            "uncertainty_mu": mu,
            "uncertainty_log_var": log_var,
        }
