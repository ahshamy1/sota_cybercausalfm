from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class ThreatIncident:
    incident_id: str
    summary: str
    mitre_tactics: List[str]


class ThreatMemory:
    """Simple vector memory for threat retrieval (RAG-ready primitive)."""

    def __init__(self) -> None:
        self._vectors: List[np.ndarray] = []
        self._incidents: List[ThreatIncident] = []

    def store(self, vector: np.ndarray, incident: ThreatIncident) -> None:
        v = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(v) + 1e-8
        self._vectors.append(v / norm)
        self._incidents.append(incident)

    def retrieve(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[ThreatIncident, float]]:
        if not self._vectors:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        q = q / (np.linalg.norm(q) + 1e-8)
        sims = np.array([float(np.dot(v, q)) for v in self._vectors], dtype=np.float32)
        idx = np.argsort(-sims)[:max(1, top_k)]
        return [(self._incidents[i], float(sims[i])) for i in idx]
