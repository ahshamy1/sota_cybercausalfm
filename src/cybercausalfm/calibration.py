from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class CalibrationReport:
    temperature: float
    ece_before: float
    ece_after: float
    nll_before: float
    nll_after: float


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy()


def expected_calibration_error(
    probs: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    n_bins: int = 15,
) -> float:
    p = _to_numpy(probs)
    y = _to_numpy(labels).astype(np.int64)

    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    acc = (pred == y).astype(np.float32)

    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (conf > lo) & (conf <= hi)
        if not np.any(mask):
            continue
        bin_acc = float(acc[mask].mean())
        bin_conf = float(conf[mask].mean())
        ece += (float(mask.mean()) * abs(bin_acc - bin_conf))
    return float(ece)


def _nll_from_logits(logits: torch.Tensor, labels: torch.Tensor, temperature: torch.Tensor | None = None) -> torch.Tensor:
    if temperature is None:
        scaled = logits
    else:
        scaled = logits / torch.clamp(temperature, min=1e-4)
    return F.cross_entropy(scaled, labels)


def fit_temperature_scaling(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 200,
    lr: float = 0.02,
) -> float:
    temperature = torch.nn.Parameter(torch.tensor(1.0, dtype=logits.dtype, device=logits.device))
    optimizer = torch.optim.Adam([temperature], lr=lr)

    for _ in range(max_iter):
        optimizer.zero_grad(set_to_none=True)
        loss = _nll_from_logits(logits, labels, temperature)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            temperature.clamp_(0.05, 10.0)

    return float(temperature.detach().cpu().item())


def calibrate_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    t = torch.tensor(temperature, dtype=logits.dtype, device=logits.device)
    return logits / torch.clamp(t, min=1e-4)


def run_calibration_pipeline(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> CalibrationReport:
    probs_before = torch.softmax(logits, dim=-1)
    ece_before = expected_calibration_error(probs_before, labels, n_bins=n_bins)
    nll_before = float(_nll_from_logits(logits, labels).detach().cpu().item())

    temperature = fit_temperature_scaling(logits, labels)
    logits_after = calibrate_logits(logits, temperature)
    probs_after = torch.softmax(logits_after, dim=-1)
    ece_after = expected_calibration_error(probs_after, labels, n_bins=n_bins)
    nll_after = float(_nll_from_logits(logits_after, labels).detach().cpu().item())

    return CalibrationReport(
        temperature=float(temperature),
        ece_before=float(ece_before),
        ece_after=float(ece_after),
        nll_before=float(nll_before),
        nll_after=float(nll_after),
    )
