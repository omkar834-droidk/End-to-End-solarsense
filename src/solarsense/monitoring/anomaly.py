"""Fault detection from the performance ratio (actual / expected power).

Expected power comes from a physics-informed model calibrated on a commissioning window.
Rules (explainable on purpose, so an O&M engineer can audit every alert):
  - inverter_outage : >= 1 h of daylight samples producing < 8 % of expected power
  - string_fault    : daily PR steps down > 7 % vs the previous 3 days and stays down
  - soiling         : slow PR decline (> 0.05 %/day) with PR > 2.5 % below reference
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DT = 0.25
GAMMA = -0.004  # typical c-Si power temperature coefficient (per degC)
FAULT_KINDS = ("inverter_outage", "string_fault")


@dataclass
class PerformanceModel:
    capacity_kw: float
    k: float
    ref_pr: float = 1.0
    gamma: float = GAMMA

    def expected(self, ghi, module_temp) -> np.ndarray:
        raw = self.k * self.capacity_kw * (np.asarray(ghi) / 1000.0) * (1 + self.gamma * (np.asarray(module_temp) - 25.0))
        return np.clip(raw, 0.0, self.capacity_kw)

    @classmethod
    def calibrate(cls, df: pd.DataFrame, capacity_kw: float, window_days: int = 21) -> "PerformanceModel":
        w = df.iloc[: window_days * 96]
        raw = capacity_kw * (w["ghi"].values / 1000.0) * (1 + GAMMA * (w["module_temp"].values - 25.0))
        ok = (w["ghi"].values >= 200) & (w["power_kw"].values < 0.95 * capacity_kw)
        k = float((raw[ok] * w["power_kw"].values[ok]).sum() / (raw[ok] ** 2).sum())
        model = cls(capacity_kw, k)
        model.ref_pr = float(np.nanmedian(daily_stats(w, model)["pr"].values))
        return model


@dataclass
class Anomaly:
    kind: str
    start: pd.Timestamp
    end: pd.Timestamp
    severity_pct: float
    lost_kwh: float
    lost_inr: float
    confidence: float
    ongoing: bool
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "severity_pct": round(self.severity_pct, 1),
            "lost_kwh": round(self.lost_kwh),
            "lost_inr": round(self.lost_inr),
            "confidence": round(self.confidence, 2),
            "ongoing": self.ongoing,
            "evidence": self.evidence,
        }


def daily_stats(df: pd.DataFrame, perf: PerformanceModel, exclude: np.ndarray | None = None,
                min_ghi: float = 250.0) -> pd.DataFrame:
    exp = perf.expected(df["ghi"].values, df["module_temp"].values)
    ok = (df["ghi"].values >= min_ghi) & (exp > 1)
    if exclude is not None:
        ok = ok & ~np.asarray(exclude)
    d = pd.DataFrame(
        {"act": np.where(ok, df["power_kw"].values, 0.0), "exp": np.where(ok, exp, 0.0), "n": ok.astype(int)},
        index=df.index,
    )
    g = d.groupby(d.index.normalize()).sum()
    pr = (g["act"] / g["exp"].where(g["exp"] > 0)).where(g["n"] >= 8)
    return pd.DataFrame({"pr": pr, "exp_kwh": g["exp"] * DT, "act_kwh": g["act"] * DT})


def _runs(flag: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    runs, i, n = [], 0, len(flag)
    while i < n:
        if flag[i]:
            j = i
            while j + 1 < n and flag[j + 1]:
                j += 1
            if j - i + 1 >= min_len:
                runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def detect(df: pd.DataFrame, perf: PerformanceModel, tariff: float, min_ghi: float = 250.0) -> list[Anomaly]:
    exp = perf.expected(df["ghi"].values, df["module_temp"].values)
    act = df["power_kw"].values
    idx = df.index
    day = (df["ghi"].values >= min_ghi) & (exp > 1)
    ratio = np.where(day, act / np.maximum(exp, 1e-9), np.nan)
    found: list[Anomaly] = []

    # 1) inverter outages (runs are taken over daylight samples, so an outage may span a night)
    pos = np.flatnonzero(day)
    outage_mask = np.zeros(len(df), dtype=bool)
    for i0, i1 in _runs(ratio[pos] < 0.08, min_len=4):
        s, e = pos[i0], pos[i1]
        outage_mask[s : e + 1] = True
        lost = float(np.clip(exp[s : e + 1] - act[s : e + 1], 0, None).sum() * DT)
        found.append(Anomaly(
            "inverter_outage", idx[s], idx[e], 100 * (1 - float(np.nanmedian(ratio[s : e + 1]))),
            lost, lost * tariff, 0.98, bool(i1 == len(pos) - 1),
            {"min_ratio": round(float(np.nanmin(ratio[s : e + 1])), 3), "daylight_hours": round((i1 - i0 + 1) * DT, 1)},
        ))

    # 2) string / partial faults: persistent step down in daily PR
    stats = daily_stats(df, perf, exclude=outage_mask, min_ghi=min_ghi)
    pr = stats["pr"].dropna()
    dates, vals = pr.index, pr.values
    events: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    i = 3
    while i < len(vals):
        prev = float(np.median(vals[i - 3 : i]))
        drop = prev - vals[i]
        if drop > 0.07 and (i + 1 >= len(vals) or vals[i + 1] < prev - 0.05):
            k = i + 1
            while k < len(vals) and not (vals[k] >= prev - 0.03 and (k + 1 >= len(vals) or vals[k + 1] >= prev - 0.03)):
                k += 1
            end_i, ongoing = k - 1, k >= len(vals)
            start_ts = dates[i]
            end_ts = dates[end_i] + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
            m = (idx >= start_ts) & (idx <= end_ts) & day & ~outage_mask
            lost = float(np.clip(exp[m] * prev - act[m], 0, None).sum() * DT)
            sev = 100 * (1 - float(np.median(vals[i : end_i + 1])) / prev)
            found.append(Anomaly(
                "string_fault", start_ts, end_ts, sev, lost, lost * tariff,
                min(0.99, 0.6 + drop * 2.5), ongoing,
                {"pr_before": round(prev, 3), "pr_after": round(float(np.median(vals[i : end_i + 1])), 3)},
            ))
            events.append((start_ts, dates[end_i]))
            i = end_i + 4
        else:
            i += 1

    # 3) soiling: slow decline on days not explained by a fault
    clean = pr.copy()
    for s_ts, e_ts in events:
        clean = clean[(clean.index < s_ts) | (clean.index > e_ts)]
    win = clean.iloc[-14:]
    if len(win) >= 10 and win.index[-1] >= dates[-1] - pd.Timedelta(days=30):
        slope = float(np.polyfit(np.arange(len(win)), win.values, 1)[0])
        level = float(win.iloc[-3:].mean())
        loss = perf.ref_pr - level
        if slope < -0.0005 and loss > 0.025:
            w = stats.loc[win.index]
            lost = float(np.clip(perf.ref_pr * w["exp_kwh"] - w["act_kwh"], 0, None).sum())
            found.append(Anomaly(
                "soiling", win.index[0], win.index[-1] + pd.Timedelta(days=1) - pd.Timedelta(minutes=15),
                100 * loss / perf.ref_pr, lost, lost * tariff, min(0.9, 0.5 + abs(slope) * 150),
                bool(win.index[-1] >= dates[-1] - pd.Timedelta(days=2)),
                {"slope_pct_per_day": round(100 * slope, 3), "pr_now": round(level, 3)},
            ))

    return sorted(found, key=lambda a: a.start)


def anomaly_mask(index: pd.DatetimeIndex, anomalies: list[Anomaly], kinds=FAULT_KINDS) -> np.ndarray:
    mask = np.zeros(len(index), dtype=bool)
    for a in anomalies:
        if a.kind in kinds:
            mask |= (index >= a.start) & (index <= a.end)
    return mask
