import importlib


def test_causal_module_imports():
    mod = importlib.import_module("cybercausalfm.causal")
    assert mod is not None


def test_causal_has_public_callable():
    mod = importlib.import_module("cybercausalfm.causal")
    public_names = [n for n in dir(mod) if not n.startswith("_")]
    public_callables = [n for n in public_names if callable(getattr(mod, n))]
    assert len(public_callables) > 0