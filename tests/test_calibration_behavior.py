import dataclasses
import math

import numpy as np
import torch

from cybercausalfm.calibration import (
    calibrate_logits,
    expected_calibration_error,
    fit_temperature_scaling,
    run_calibration_pipeline,
)


def test_calibrate_logits_temperature_one_identity():
    logits = torch.tensor([[2.0, 0.5], [0.1, 1.9]], dtype=torch.float32)
    out = calibrate_logits(logits, temperature=1.0)
    assert torch.allclose(out, logits, atol=1e-6)


def test_calibrate_logits_changes_with_temperature():
    logits = torch.tensor([[2.0, 0.5], [0.1, 1.9]], dtype=torch.float32)
    out = calibrate_logits(logits, temperature=2.0)
    assert out.shape == logits.shape
    assert not torch.allclose(out, logits, atol=1e-6)


def test_expected_calibration_error_perfect_predictions_near_zero():
    # Near-perfect confidence aligned with labels
    probs = np.array(
        [
            [0.99, 0.01],
            [0.01, 0.99],
            [0.98, 0.02],
            [0.02, 0.98],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 1, 0, 1], dtype=np.int64)

    ece = expected_calibration_error(probs, labels, n_bins=10)
    assert isinstance(ece, float)
    assert math.isfinite(ece)
    assert 0.0 <= ece <= 1.0
    assert ece < 0.1


def test_fit_temperature_scaling_returns_positive_finite():
    torch.manual_seed(7)
    logits = torch.randn(128, 3)
    labels = torch.randint(0, 3, (128,))

    t = fit_temperature_scaling(logits, labels, max_iter=50, lr=0.02)
    assert isinstance(t, float)
    assert math.isfinite(t)
    assert t > 0.0


def test_run_calibration_pipeline_returns_dataclass_like_report():
    torch.manual_seed(11)
    logits = torch.randn(64, 4)
    labels = torch.randint(0, 4, (64,))

    report = run_calibration_pipeline(logits, labels, n_bins=15)

    # Behavioral check: report object with structured fields
    assert report is not None
    if dataclasses.is_dataclass(report):
        assert len(dataclasses.asdict(report)) > 0
    else:
        assert hasattr(report, "__dict__")
        assert len(vars(report)) > 0