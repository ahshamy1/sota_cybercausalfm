# CyberCausalFM (SOTA Scaffold)

![CI](https://github.com/ahshamy1/sota_cybercausalfm/actions/workflows/ci.yml/badge.svg)

This project is a runnable scaffold generated from the concepts and snippets extracted from `CS.pdf`, then upgraded into a practical SOTA-style architecture:

- Cyber foundation pretraining objectives:
  - Masked Cyber Event Modeling
  - Next Event Prediction
  - Cross-modal Contrastive Alignment
- MITRE ATT&CK mapping engine
- Threat memory retrieval module
- Uncertainty-aware autonomous response engine
- Digital twin attack simulation loop

## Quick start

```powershell
cd C:\P26\sota_cybercausalfm
pip install -r requirements.txt
python scripts\train_demo.py
```

## Full SOTA Upgrade Run

```powershell
cd C:\P26\sota_cybercausalfm
python scripts\run_sota_upgrade.py
```

This command runs:

- scaling-law experiment (`artifacts/scaling_law_results.csv`)
- calibration pipeline (`artifacts/calibration_report.json`)
- causal intervention ranking (`artifacts/causal_interventions.json`)
- combined summary (`artifacts/sota_upgrade_summary.json`)

## Project structure

- `src/cybercausalfm/model.py`: multimodal model with temporal encoder + graph branch + uncertainty head
- `src/cybercausalfm/objectives.py`: self-supervised pretraining losses
- `src/cybercausalfm/mitre.py`: ATT&CK mapping extracted from PDF and expanded
- `src/cybercausalfm/memory.py`: threat memory retrieval
- `src/cybercausalfm/decision.py`: risk+uncertainty response policy
- `src/cybercausalfm/digital_twin.py`: synthetic enterprise environment simulator
- `scripts/train_demo.py`: end-to-end demo run
- `scripts/run_sota_upgrade.py`: one-command scaling + calibration + causal upgrade run

## Notes

This is a production-oriented starting point, not a final benchmarked paper implementation.
To make it fully publishable, add:

1. Multi-dataset scaling-law experiments
2. Real graph causal discovery and intervention tests
3. Calibration error evaluation and post-hoc calibration
4. Distributed/federated training and full MLOps deployment