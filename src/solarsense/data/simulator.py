"""Synthetic utility-scale solar plant.

Generates 15-minute weather, soiling, NWP-style forecast inputs and injected faults
(with ground-truth labels), so detection and forecasting can be tested end to end.
Replace `simulate()` with a SCADA/CSV loader that returns the same columns for real data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

STEPS_PER_DAY = 96
DT_HOURS = 0.25


@dataclass(frozen=True)
class FaultEvent:
    kind: str  # "inverter_outage" | "string_fault"
    start_day: float  # days from simulation start
    end_day: float
    severity: float  # fraction of power lost (1.0 = total outage)


DEFAULT_FAULTS = (
    FaultEvent("inverter_outage", 40.30, 40.70, 1.0),
    FaultEvent("inverter_outage", 112.40, 113.20, 1.0),
    FaultEvent("string_fault", 128.00, 400.0, 0.22),
)
DEFAULT_CLEANING_DAYS = (0, 38, 77)


def solar_cos_zenith(index: pd.DatetimeIndex, lat: float, lon: float) -> np.ndarray:
    doy = index.dayofyear.values
    hour = index.hour.values + index.minute.values / 60.0
    decl = np.radians(23.45 * np.sin(2 * np.pi * (284 + doy) / 365))
    solar_time = hour + (lon - 82.5) / 15.0  # index is IST (UTC+5:30, meridian 82.5E)
    h = np.radians(15.0 * (solar_time - 12.0))
    lat_r = np.radians(lat)
    cosz = np.sin(lat_r) * np.sin(decl) + np.cos(lat_r) * np.cos(decl) * np.cos(h)
    return np.clip(cosz, 0.0, None)


def clearsky_ghi(cos_z: np.ndarray) -> np.ndarray:
    return 1050.0 * np.power(cos_z, 1.15)


def simulate(
    start: str = "2026-01-01",
    days: int = 150,
    capacity_kw: float = 5000.0,
    lat: float = 18.52,
    lon: float = 73.86,
    seed: int = 42,
    faults: tuple[FaultEvent, ...] = DEFAULT_FAULTS,
    cleaning_days: tuple[int, ...] = DEFAULT_CLEANING_DAYS,
    dc_ac_ratio: float = 1.35,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * STEPS_PER_DAY, freq="15min")
    n = len(idx)
    cosz = solar_cos_zenith(idx, lat, lon)
    cs = clearsky_ghi(cosz)
    doy = idx.dayofyear.values
    hour = idx.hour.values + idx.minute.values / 60.0

    # Cloud process: AR(1) with a pre-monsoon ramp
    eps = rng.normal(0, 0.18, n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.985 * x[t - 1] + eps[t]
    season = np.clip((doy - 80) / 100.0, 0, 1)
    cloud = 1.0 / (1.0 + np.exp(-(1.1 * x - 1.2 + 1.2 * season)))
    kt = 1.0 - 0.8 * cloud**1.3
    ghi = np.clip(cs * kt * (1 + rng.normal(0, 0.02, n)), 0, None)

    seasonal = 22 + 8 * np.clip((doy - 20) / 100.0, 0, 1)
    ambient = seasonal + 7 * np.sin(2 * np.pi * (hour - 9) / 24) - 4 * cloud + rng.normal(0, 0.6, n)
    module_temp = ambient + 0.028 * ghi

    # Soiling: grows daily, reset by cleaning or very cloudy (rainy) days
    day_idx = np.arange(n) // STEPS_PER_DAY
    daily_cloud = pd.Series(cloud).groupby(day_idx).mean().values
    soil_day = np.zeros(days)
    s = 0.0
    for d in range(days):
        if d in cleaning_days or daily_cloud[d] > 0.8:
            s = 0.0
        else:
            s = min(s + 0.0015 * rng.uniform(0.7, 1.3), 0.15)
        soil_day[d] = s
    soil = soil_day[day_idx]

    temp_factor = 1 - 0.004 * (module_temp - 25.0)
    dc = capacity_kw * dc_ac_ratio * (ghi / 1000.0) * 0.84 * temp_factor * (1 - soil)
    power = np.clip(dc * (1 + rng.normal(0, 0.012, n)), 0, capacity_kw)

    label = np.full(n, "none", dtype=object)
    pos = np.arange(n)
    for f in faults:
        m = (pos >= f.start_day * STEPS_PER_DAY) & (pos < f.end_day * STEPS_PER_DAY)
        power = np.where(m, power * (1 - f.severity), power)
        label[m] = f.kind

    # NWP-style forecast inputs (what a day-ahead scheduler actually has)
    def smooth_noise(sd: float) -> np.ndarray:
        return pd.Series(rng.normal(0, sd, n)).rolling(8, center=True, min_periods=1).mean().values

    kt_fc = np.clip(kt + smooth_noise(0.60), 0.05, 1.0)
    return pd.DataFrame(
        {
            "ghi": ghi,
            "ambient_temp": ambient,
            "module_temp": module_temp,
            "cloud_cover": cloud,
            "clearsky_ghi": cs,
            "ghi_fc": cs * kt_fc,
            "temp_fc": ambient + rng.normal(0, 1.5, n),
            "cloud_fc": np.clip(cloud + smooth_noise(0.60), 0, 1),
            "power_kw": power,
            "fault_label": label,
            "soiling_loss": soil,
        },
        index=idx,
    )
