from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class CausalEdge:
    source: str
    target: str
    weight: float


class CausalAttackGraph:
    """Simple SCM-like attack chain graph with intervention support."""

    def __init__(self) -> None:
        self.nodes: List[str] = [
            "Reconnaissance",
            "Initial Access",
            "Credential Access",
            "Privilege Escalation",
            "Lateral Movement",
            "Exfiltration",
        ]
        self.edges: Dict[Tuple[str, str], float] = {
            ("Reconnaissance", "Initial Access"): 0.55,
            ("Initial Access", "Credential Access"): 0.62,
            ("Credential Access", "Privilege Escalation"): 0.68,
            ("Privilege Escalation", "Lateral Movement"): 0.72,
            ("Lateral Movement", "Exfiltration"): 0.78,
        }

    def chain_risk(self, start_risk: float = 0.4) -> float:
        risk = float(start_risk)
        for src, dst in zip(self.nodes[:-1], self.nodes[1:]):
            w = self.edges.get((src, dst), 0.0)
            risk = min(1.0, risk + (1.0 - risk) * w)
        return risk

    def intervention(self, block_node: str, start_risk: float = 0.4) -> dict[str, float | str]:
        block = block_node.strip()
        base = self.chain_risk(start_risk=start_risk)

        risk = float(start_risk)
        for src, dst in zip(self.nodes[:-1], self.nodes[1:]):
            if src == block or dst == block:
                # Blocking a node cuts the chain and reduces propagation.
                risk *= 0.7
                break
            w = self.edges.get((src, dst), 0.0)
            risk = min(1.0, risk + (1.0 - risk) * w)

        return {
            "blocked_node": block,
            "base_risk": base,
            "post_intervention_risk": risk,
            "risk_reduction": max(0.0, base - risk),
        }

    def rank_best_interventions(self, start_risk: float = 0.4) -> List[dict[str, float | str]]:
        scored = [self.intervention(n, start_risk=start_risk) for n in self.nodes]
        scored.sort(key=lambda x: float(x["risk_reduction"]), reverse=True)
        return scored
