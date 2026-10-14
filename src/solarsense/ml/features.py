import numpy as np
import pandas as pd

FEATURES = ["tod_sin", "tod_cos", "clearsky_ghi", "ghi_fc", "temp_fc", "cloud_fc", "kt_fc"]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Day-ahead features: calendar + forecast weather only (no same-day actuals)."""
    tod = df.index.hour.values + df.index.minute.values / 60.0
    cs = df["clearsky_ghi"].values
    out = pd.DataFrame(index=df.index)
    out["tod_sin"] = np.sin(2 * np.pi * tod / 24)
    out["tod_cos"] = np.cos(2 * np.pi * tod / 24)
    out["clearsky_ghi"] = cs
    out["ghi_fc"] = df["ghi_fc"].values
    out["temp_fc"] = df["temp_fc"].values
    out["cloud_fc"] = df["cloud_fc"].values
    out["kt_fc"] = np.where(cs > 20, np.clip(df["ghi_fc"].values / np.maximum(cs, 1.0), 0, 1.2), 0.0)
    return out[FEATURES]
