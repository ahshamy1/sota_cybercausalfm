from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class TwinEvent:
    event_type: str
    risk: float
    payload: Dict[str, float]


class CyberDigitalTwin:
    """Minimal digital twin simulator for attack progression experiments."""

    ATTACK_CHAIN: List[str] = [
        "reconnaissance",
        "phishing",
        "credential theft",
        "privilege escalation",
        "lateral movement",
        "exfiltration",
    ]

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def generate_sequence(self, length: int = 32) -> List[TwinEvent]:
        events: List[TwinEvent] = []
        chain_pos = 0
        for _ in range(length):
            if self.rng.random() < 0.2:
                chain_pos = min(chain_pos + 1, len(self.ATTACK_CHAIN) - 1)
            event_type = self.ATTACK_CHAIN[chain_pos]
            base_risk = (chain_pos + 1) / len(self.ATTACK_CHAIN)
            risk = float(np.clip(base_risk + self.rng.normal(0.0, 0.08), 0.01, 0.99))
            payload = {
                "bytes_out": float(self.rng.uniform(100.0, 20000.0)),
                "failed_logins": float(self.rng.integers(0, 12)),
                "new_processes": float(self.rng.integers(0, 20)),
            }
            events.append(TwinEvent(event_type=event_type, risk=risk, payload=payload))
        return events
