"""Day-ahead probabilistic forecast: P10 / P50 / P90 via quantile gradient boosting."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from solarsense.ml.features import FEATURES, make_features

QUANTILES = (0.1, 0.5, 0.9)
STEPS_PER_DAY = 96


class QuantileForecaster:
    def __init__(self, capacity_kw: float, seed: int = 0):
        self.capacity_kw = capacity_kw
        self.seed = seed
        self.models: dict[float, HistGradientBoostingRegressor] = {}
        self.meta: dict = {}

    def fit(self, df: pd.DataFrame, exclude: np.ndarray | None = None) -> "QuantileForecaster":
        day = df["clearsky_ghi"].values > 0
        keep = day.copy()
        if exclude is not None:
            keep &= ~np.asarray(exclude)
        X = make_features(df)[keep]
        y = df["power_kw"].values[keep] / self.capacity_kw
        for q in QUANTILES:
            self.models[q] = HistGradientBoostingRegressor(
                loss="quantile", quantile=q, max_iter=200, learning_rate=0.06,
                max_leaf_nodes=20, min_samples_leaf=50, random_state=self.seed,
            ).fit(X, y)
        self.meta = {
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "train_start": str(df.index[0]),
            "train_end": str(df.index[-1]),
            "train_rows": int(keep.sum()),
            "excluded_fault_rows": int((day & ~keep).sum()),
            "features": FEATURES,
        }
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = make_features(df)
        raw = np.column_stack([self.models[q].predict(X) for q in QUANTILES])
        raw = np.clip(np.sort(raw, axis=1), 0, 1) * self.capacity_kw  # keep quantiles ordered
        raw[df["clearsky_ghi"].values <= 0] = 0.0
        return pd.DataFrame(raw, index=df.index, columns=["p10", "p50", "p90"])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "QuantileForecaster":
        return joblib.load(path)


def evaluate(df: pd.DataFrame, pred: pd.DataFrame, capacity_kw: float,
             exclude: np.ndarray | None = None) -> dict:
    """Daylight-only metrics; known-fault rows are excluded so faults do not blame the model."""
    keep = df["clearsky_ghi"].values > 0
    persist = df["power_kw"].shift(STEPS_PER_DAY).values  # yesterday, same time
    if exclude is not None:
        keep = keep & ~np.asarray(exclude)
    a = df["power_kw"].values
    p10, p50, p90 = (pred[c].values for c in ("p10", "p50", "p90"))
    has_p = keep & ~np.isnan(persist)
    return {
        "rows": int(keep.sum()),
        "nmae_pct": round(float(100 * np.mean(np.abs(a[keep] - p50[keep])) / capacity_kw), 2),
        "persistence_nmae_pct": round(float(100 * np.mean(np.abs(a[has_p] - persist[has_p])) / capacity_kw), 2)
        if has_p.any() else None,
        "bias_pct": round(float(100 * np.mean(p50[keep] - a[keep]) / capacity_kw), 2),
        "coverage_p10_p90_pct": round(float(100 * np.mean((a[keep] >= p10[keep]) & (a[keep] <= p90[keep]))), 1),
    }


def evaluate_rows(df: pd.DataFrame, pred: pd.DataFrame, capacity_kw: float, rows: np.ndarray) -> dict:
    """Metrics on an arbitrary row selection (daylight only). Used for non-contiguous healthy windows."""
    keep = np.asarray(rows) & (df["clearsky_ghi"].values > 0)
    a = df["power_kw"].values
    p10, p50, p90 = (pred[c].values for c in ("p10", "p50", "p90"))
    if not keep.any():
        return {"rows": 0, "nmae_pct": None, "persistence_nmae_pct": None, "bias_pct": None, "coverage_p10_p90_pct": None}
    return {
        "rows": int(keep.sum()),
        "nmae_pct": round(float(100 * np.mean(np.abs(a[keep] - p50[keep])) / capacity_kw), 2),
        "persistence_nmae_pct": None,
        "bias_pct": round(float(100 * np.mean(p50[keep] - a[keep]) / capacity_kw), 2),
        "coverage_p10_p90_pct": round(float(100 * np.mean((a[keep] >= p10[keep]) & (a[keep] <= p90[keep]))), 1),
    }
