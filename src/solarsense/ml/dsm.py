"""Illustrative Deviation Settlement (DSM) cost model.

Generators that schedule day-ahead are charged for deviation beyond a tolerance band.
The rules here are SIMPLIFIED and configurable. They are not the exact regulatory formula.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DT_HOURS = 0.25
STEPS_PER_DAY = 96


def penalty(actual_kw, scheduled_kw, capacity_kw: float, tariff: float,
            tolerance_pct: float, multiplier: float = 1.0) -> np.ndarray:
    band = capacity_kw * tolerance_pct / 100.0
    excess = np.clip(np.abs(np.asarray(actual_kw) - np.asarray(scheduled_kw)) - band, 0, None)
    return excess * DT_HOURS * tariff * multiplier


def compare_schedules(df: pd.DataFrame, pred: pd.DataFrame, capacity_kw: float, tariff: float,
                      tolerance_pct: float, multiplier: float = 1.0,
                      exclude: np.ndarray | None = None) -> dict:
    """Schedule at model P50 vs a persistence baseline (yesterday's output)."""
    persist = df["power_kw"].shift(STEPS_PER_DAY)
    valid = (df["clearsky_ghi"].values > 0) & persist.notna().values
    if exclude is not None:
        valid = valid & ~np.asarray(exclude)
    act = df["power_kw"].values
    model = penalty(act, pred["p50"].values, capacity_kw, tariff, tolerance_pct, multiplier)
    base = penalty(act, persist.fillna(0).values, capacity_kw, tariff, tolerance_pct, multiplier)
    daily = pd.DataFrame(
        {"model": np.where(valid, model, 0.0), "persistence": np.where(valid, base, 0.0)},
        index=df.index,
    ).groupby(df.index.normalize()).sum()
    m, b = float(daily["model"].sum()), float(daily["persistence"].sum())
    return {
        "model_penalty_inr": round(m),
        "persistence_penalty_inr": round(b),
        "savings_inr": round(b - m),
        "reduction_pct": round(100 * (b - m) / b, 1) if b > 0 else 0.0,
        "days": int(len(daily)),
        "daily": [
            {"date": str(d.date()), "model": round(float(r["model"])), "persistence": round(float(r["persistence"]))}
            for d, r in daily.iterrows()
        ],
        "assumptions": {"tariff_inr_per_kwh": tariff, "tolerance_pct": tolerance_pct, "multiplier": multiplier},
    }
