from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DecisionResult:
    action: str
    risk: float
    confidence: float
    reason: str


def decide_action(risk: float, confidence: float) -> DecisionResult:
    r = float(max(0.0, min(1.0, risk)))
    c = float(max(0.0, min(1.0, confidence)))

    if r > 0.9 and c > 0.8:
        return DecisionResult(
            action="AUTO_ISOLATE_BLOCK_TRAFFIC",
            risk=r,
            confidence=c,
            reason="Critical risk with high confidence.",
        )
    if r > 0.7:
        return DecisionResult(
            action="LIMIT_ACCESS_MONITOR",
            risk=r,
            confidence=c,
            reason="High risk but confidence or severity not maximal.",
        )
    return DecisionResult(
        action="LOG_ONLY_MONITOR",
        risk=r,
        confidence=c,
        reason="Risk below escalation threshold.",
    )
