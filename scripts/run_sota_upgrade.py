from __future__ import annotations

import csv
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from cybercausalfm.causal import CausalAttackGraph
from cybercausalfm.calibration import run_calibration_pipeline
from cybercausalfm.model import CyberCausalFM
from cybercausalfm.scaling import run_scaling_law_experiment, scaling_points_to_rows


def _ensure_artifact_dir() -> str:
    out_dir = os.path.join(ROOT, "artifacts")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_scaling_stage(out_dir: str) -> dict:
    points = run_scaling_law_experiment(seed=13)
    rows = scaling_points_to_rows(points)
    csv_path = os.path.join(out_dir, "scaling_law_results.csv")
    _write_csv(csv_path, rows)

    best = max(rows, key=lambda x: x["zero_day_proxy_acc"])
    summary = {
        "stage": "scaling_laws",
        "num_points": len(rows),
        "best_zero_day_proxy_acc": best["zero_day_proxy_acc"],
        "best_config": {
            "event_dim": best["event_dim"],
            "n_layers": best["n_layers"],
            "n_heads": best["n_heads"],
            "params": best["params"],
        },
        "csv": csv_path,
    }
    print(f"[scaling] points={len(rows)} best_zd={best['zero_day_proxy_acc']:.4f}")
    return summary


def run_calibration_stage(out_dir: str) -> dict:
    torch.manual_seed(29)

    model = CyberCausalFM()
    tokens = torch.randint(0, 128, (256, 24))
    graph_x = torch.randn(256, 12, 32)
    adj = torch.rand(256, 12, 12)
    adj = (adj + adj.transpose(1, 2)) / 2.0
    labels = torch.randint(0, 6, (256,))

    with torch.no_grad():
        logits = model(tokens, graph_x, adj)["class_logits"]

    report = run_calibration_pipeline(logits, labels, n_bins=15)
    payload = {
        "stage": "calibration",
        "temperature": report.temperature,
        "ece_before": report.ece_before,
        "ece_after": report.ece_after,
        "nll_before": report.nll_before,
        "nll_after": report.nll_after,
    }
    json_path = os.path.join(out_dir, "calibration_report.json")
    _write_json(json_path, payload)
    print(
        "[calibration] "
        f"temp={report.temperature:.4f} "
        f"ece_before={report.ece_before:.4f} ece_after={report.ece_after:.4f}"
    )
    payload["json"] = json_path
    return payload


def run_causal_stage(out_dir: str) -> dict:
    graph = CausalAttackGraph()
    ranking = graph.rank_best_interventions(start_risk=0.42)
    payload = {
        "stage": "causal_intervention",
        "base_chain_risk": graph.chain_risk(start_risk=0.42),
        "top_interventions": ranking[:3],
        "all_interventions": ranking,
    }
    json_path = os.path.join(out_dir, "causal_interventions.json")
    _write_json(json_path, payload)
    print(
        "[causal] "
        f"base_risk={payload['base_chain_risk']:.4f} "
        f"best={payload['top_interventions'][0]['blocked_node']}"
    )
    payload["json"] = json_path
    return payload


def main() -> None:
    out_dir = _ensure_artifact_dir()
    scaling = run_scaling_stage(out_dir)
    calibration = run_calibration_stage(out_dir)
    causal = run_causal_stage(out_dir)

    summary = {
        "scaling": scaling,
        "calibration": calibration,
        "causal": causal,
    }
    summary_path = os.path.join(out_dir, "sota_upgrade_summary.json")
    _write_json(summary_path, summary)

    print("\n=== SOTA UPGRADE COMPLETE ===")
    print(f"Artifacts dir: {out_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
