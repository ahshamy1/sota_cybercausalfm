import json
import subprocess
import sys
from pathlib import Path


def test_run_sota_upgrade_smoke():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "run_sota_upgrade.py"

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    assert result.returncode == 0, (
        f"run_sota_upgrade.py failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    artifacts = repo_root / "artifacts"
    expected = [
        artifacts / "calibration_report.json",
        artifacts / "causal_interventions.json",
        artifacts / "scaling_law_results.csv",
        artifacts / "sota_upgrade_summary.json",
    ]

    for path in expected:
        assert path.exists(), f"Missing artifact: {path}"

    with open(artifacts / "calibration_report.json", "r", encoding="utf-8") as f:
        assert isinstance(json.load(f), (dict, list))

    with open(artifacts / "causal_interventions.json", "r", encoding="utf-8") as f:
        assert isinstance(json.load(f), (dict, list))

    with open(artifacts / "sota_upgrade_summary.json", "r", encoding="utf-8") as f:
        assert isinstance(json.load(f), (dict, list))

    csv_text = (artifacts / "scaling_law_results.csv").read_text(encoding="utf-8").strip()
    assert len(csv_text) > 0
    assert "," in csv_text.splitlines()[0]