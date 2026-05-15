from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

# Extracted seed mapping from CS.pdf and expanded for practical SOC coverage.
MITRE_MAP: Dict[str, str] = {
    "phishing": "Initial Access",
    "credential theft": "Credential Access",
    "lateral movement": "Lateral Movement",
    "exfiltration": "Exfiltration",
    "reconnaissance": "Reconnaissance",
    "privilege escalation": "Privilege Escalation",
    "command and control": "Command and Control",
    "defense evasion": "Defense Evasion",
    "impact": "Impact",
}


def map_attack(event: str) -> str:
    key = (event or "").strip().lower()
    return MITRE_MAP.get(key, "Unknown")


@dataclass
class MitreAnnotation:
    event: str
    tactic: str


def annotate_events(events: List[str]) -> List[MitreAnnotation]:
    return [MitreAnnotation(event=e, tactic=map_attack(e)) for e in events]
