import dataclasses

from cybercausalfm.decision import DecisionResult, decide_action


def _to_dict(obj):
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if hasattr(obj, "__dict__"):
        return vars(obj)
    return {"value": obj}


def test_decide_action_returns_decision_result():
    result = decide_action(risk=0.8, confidence=0.9)
    assert isinstance(result, DecisionResult)


def test_decide_action_returns_structured_output():
    result = decide_action(risk=0.5, confidence=0.5)
    data = _to_dict(result)

    assert isinstance(data, dict)
    assert len(data) > 0


def test_decide_action_is_deterministic_for_same_input():
    result1 = decide_action(risk=0.7, confidence=0.8)
    result2 = decide_action(risk=0.7, confidence=0.8)

    assert _to_dict(result1) == _to_dict(result2)