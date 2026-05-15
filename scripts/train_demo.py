from __future__ import annotations

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from cybercausalfm.decision import decide_action
from cybercausalfm.digital_twin import CyberDigitalTwin
from cybercausalfm.memory import ThreatIncident, ThreatMemory
from cybercausalfm.mitre import annotate_events
from cybercausalfm.model import CyberCausalFM
from cybercausalfm.objectives import (
    contrastive_alignment_loss,
    masked_event_modeling_loss,
    next_event_prediction_loss,
)


def build_batch(batch_size: int = 16, seq_len: int = 24, vocab_size: int = 128):
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = tokens.clone()
    mask = torch.rand(batch_size, seq_len) < 0.15
    masked_tokens = tokens.clone()
    masked_tokens[mask] = 0

    graph_x = torch.randn(batch_size, 12, 32)
    adj = torch.rand(batch_size, 12, 12)
    adj = (adj + adj.transpose(1, 2)) / 2.0

    next_labels = torch.randint(0, vocab_size, (batch_size,))
    class_labels = torch.randint(0, 6, (batch_size,))

    return masked_tokens, labels, mask, graph_x, adj, next_labels, class_labels


def run_demo() -> None:
    torch.manual_seed(7)
    np.random.seed(7)

    model = CyberCausalFM()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

    for step in range(1, 11):
        model.train()
        batch = build_batch()
        masked_tokens, labels, mask, graph_x, adj, next_labels, class_labels = batch

        out = model(masked_tokens, graph_x, adj)
        l_mcm = masked_event_modeling_loss(out["token_logits"], labels, mask)
        l_nap = next_event_prediction_loss(out["next_event_logits"], next_labels)
        # Contrastive proxy: align fusion with itself under tiny perturbation.
        z1 = out["fusion"]
        z2 = out["fusion"] + 0.01 * torch.randn_like(out["fusion"])
        l_align = contrastive_alignment_loss(z1, z2)
        l_cls = torch.nn.functional.cross_entropy(out["class_logits"], class_labels)

        loss = l_mcm + l_nap + 0.2 * l_align + l_cls

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 2 == 0:
            print(f"step={step:02d} loss={loss.item():.4f} mcm={l_mcm.item():.4f} nap={l_nap.item():.4f}")

    # Digital twin + MITRE + memory + decision demonstration.
    twin = CyberDigitalTwin(seed=123)
    seq = twin.generate_sequence(length=12)
    events = [e.event_type for e in seq]
    annotations = annotate_events(events)

    memory = ThreatMemory()
    for i in range(3):
        vec = np.random.randn(192).astype(np.float32)
        incident = ThreatIncident(
            incident_id=f"INC-{100+i}",
            summary=f"Synthetic incident {i}",
            mitre_tactics=list({a.tactic for a in annotations[:3]}),
        )
        memory.store(vec, incident)

    query = np.random.randn(192).astype(np.float32)
    retrieved = memory.retrieve(query, top_k=2)

    risk = float(np.mean([e.risk for e in seq]))
    confidence = 0.86
    decision = decide_action(risk=risk, confidence=confidence)

    print("\n--- SOC OUTPUT ---")
    print("Recent MITRE tactics:", [a.tactic for a in annotations[-4:]])
    print("Retrieved incidents:", [(r.incident_id, round(score, 3)) for r, score in retrieved])
    print("Risk:", round(risk, 3), "Confidence:", confidence)
    print("Decision:", decision.action, "|", decision.reason)


if __name__ == "__main__":
    run_demo()
