import inspect
import math

from cybercausalfm.causal import CausalAttackGraph, CausalEdge


def _build_edge(src: str, dst: str, p: float):
    sig = inspect.signature(CausalEdge)
    kwargs = {}
    for name, param in sig.parameters.items():
        lname = name.lower()

        if "src" in lname or "source" in lname or lname in {"u", "from_node", "from_"}:
            kwargs[name] = src
        elif "dst" in lname or "target" in lname or lname in {"v", "to_node", "to"}:
            kwargs[name] = dst
        elif any(k in lname for k in ["prob", "weight", "risk", "likelihood", "strength"]):
            kwargs[name] = float(p)
        elif param.default is not inspect._empty:
            kwargs[name] = param.default
        elif param.annotation in (str, "str"):
            kwargs[name] = "x"
        elif param.annotation in (int, "int"):
            kwargs[name] = 0
        else:
            kwargs[name] = 0.5
    return CausalEdge(**kwargs)


def _build_graph():
    edges = [
        _build_edge("InitialAccess", "Execution", 0.6),
        _build_edge("Execution", "PrivilegeEscalation", 0.5),
        _build_edge("PrivilegeEscalation", "Impact", 0.7),
    ]

    sig = inspect.signature(CausalAttackGraph)
    kwargs = {}
    for name, param in sig.parameters.items():
        lname = name.lower()
        if "edge" in lname:
            kwargs[name] = edges
        elif "node" in lname:
            kwargs[name] = ["InitialAccess", "Execution", "PrivilegeEscalation", "Impact"]
        elif param.default is not inspect._empty:
            kwargs[name] = param.default
        else:
            # Fallback for required params
            kwargs[name] = edges
    return CausalAttackGraph(**kwargs)


def test_chain_risk_returns_finite_value():
    graph = _build_graph()
    risk = graph.chain_risk(start_risk=0.4)
    assert isinstance(risk, float)
    assert math.isfinite(risk)


def test_intervention_returns_structured_dict():
    graph = _build_graph()
    result = graph.intervention(block_node="Execution", start_risk=0.4)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_rank_best_interventions_returns_list_of_dicts():
    graph = _build_graph()
    ranked = graph.rank_best_interventions(start_risk=0.4)
    assert isinstance(ranked, list)
    if ranked:
        assert isinstance(ranked[0], dict)