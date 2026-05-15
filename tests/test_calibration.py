import importlib


def test_calibration_module_imports():
    mod = importlib.import_module("cybercausalfm.calibration")
    assert mod is not None


def test_calibration_has_public_callable():
    mod = importlib.import_module("cybercausalfm.calibration")
    public_names = [n for n in dir(mod) if not n.startswith("_")]
    public_callables = [n for n in public_names if callable(getattr(mod, n))]
    assert len(public_callables) > 0