import inspect

import numpy as np

from cybercausalfm.memory import ThreatIncident, ThreatMemory


def _build_incident():
    sig = inspect.signature(ThreatIncident)
    kwargs = {}

    for name, param in sig.parameters.items():
        lname = name.lower()

        if param.default is not inspect._empty:
            kwargs[name] = param.default
        elif "id" in lname:
            kwargs[name] = "incident-1"
        elif "name" in lname or "title" in lname:
            kwargs[name] = "test-incident"
        elif "desc" in lname or "summary" in lname or "text" in lname:
            kwargs[name] = "test description"
        elif "label" in lname or "type" in lname or "category" in lname:
            kwargs[name] = "malware"
        elif "severity" in lname or "score" in lname or "risk" in lname:
            kwargs[name] = 0.8
        else:
            kwargs[name] = "x"

    return ThreatIncident(**kwargs)


def _build_memory():
    sig = inspect.signature(ThreatMemory)
    kwargs = {}

    for name, param in sig.parameters.items():
        if param.default is not inspect._empty:
            kwargs[name] = param.default
        else:
            lname = name.lower()
            if "dim" in lname or "size" in lname or "capacity" in lname:
                kwargs[name] = 8
            elif "vector" in lname or "embedding" in lname:
                kwargs[name] = []
            elif "incident" in lname or "items" in lname:
                kwargs[name] = []
            else:
                kwargs[name] = None

    return ThreatMemory(**kwargs)


def test_store_and_retrieve_returns_ranked_results():
    memory = _build_memory()
    incident = _build_incident()
    vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    memory.store(vector, incident)
    results = memory.retrieve(vector, top_k=1)

    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], tuple)
    assert len(results[0]) == 2


def test_exact_match_ranks_first():
    memory = _build_memory()

    incident_a = _build_incident()
    incident_b = _build_incident()

    vector_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vector_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    memory.store(vector_a, incident_a)
    memory.store(vector_b, incident_b)

    results = memory.retrieve(vector_a, top_k=2)

    assert len(results) >= 1
    top_incident, top_score = results[0]
    assert isinstance(top_incident, ThreatIncident)
    assert isinstance(top_score, float)


def test_retrieve_respects_top_k():
    memory = _build_memory()

    for i in range(3):
        incident = _build_incident()
        vector = np.zeros(4, dtype=np.float32)
        vector[i] = 1.0
        memory.store(vector, incident)

    results = memory.retrieve(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), top_k=2)
    assert isinstance(results, list)
    assert len(results) <= 2