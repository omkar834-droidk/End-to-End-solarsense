"""Data and performance drift: Population Stability Index + error-ratio trigger."""
from __future__ import annotations

import numpy as np
import pandas as pd

WATCH = ("ghi_fc", "temp_fc", "cloud_fc")


def psi(ref, cur, bins: int = 10, eps: float = 1e-4) -> float:
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.clip(np.histogram(ref, edges)[0] / len(ref), eps, None)
    c = np.clip(np.histogram(cur, edges)[0] / len(cur), eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def drift_report(ref: pd.DataFrame, cur: pd.DataFrame, nmae_ref: float | None = None,
                 nmae_cur: float | None = None, cols=WATCH) -> dict:
    ref_d, cur_d = ref[ref["clearsky_ghi"] > 0], cur[cur["clearsky_ghi"] > 0]
    scores = {c: round(psi(ref_d[c], cur_d[c]), 3) for c in cols}
    worst = max(scores.values())
    ratio = round(nmae_cur / nmae_ref, 2) if nmae_ref and nmae_cur else None
    retrain = worst > 0.25 or (ratio is not None and ratio > 1.5)
    status = "retrain" if retrain else ("watch" if worst > 0.10 else "ok")
    return {
        "status": status,
        "retrain_recommended": retrain,
        "psi": scores,
        "max_psi": worst,
        "nmae_ref_pct": nmae_ref,
        "nmae_current_pct": nmae_cur,
        "error_ratio": ratio,
        "thresholds": {"psi_watch": 0.10, "psi_retrain": 0.25, "error_ratio_retrain": 1.5},
    }
