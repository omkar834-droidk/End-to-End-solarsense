"""Rebuild data, detect faults, retrain the forecaster and print the validation report.

    python scripts/train.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solarsense.service import PlantService  # noqa: E402

if __name__ == "__main__":
    svc = PlantService()
    if svc.model_path.exists():
        svc.model_path.unlink()
    svc.ensure_ready()
    print("Faults found in the training pipeline:")
    for a in svc.anomalies:
        print(f"  {a.kind:16s} {a.start:%Y-%m-%d} -> {a.end:%Y-%m-%d}  loss {a.severity_pct:5.1f}%  INR {a.lost_inr:,.0f}")
    print("\nValidation:", json.dumps(svc.holdout))
    print("DSM (illustrative):", json.dumps({k: v for k, v in svc.dsm(60).items() if k != "daily"}))
    print("Drift:", json.dumps(svc.drift()))
    print(f"\nModel saved to {svc.model_path}")
