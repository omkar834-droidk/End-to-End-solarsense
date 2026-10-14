import numpy as np
import pandas as pd

from solarsense.ml.dsm import penalty
from solarsense.ml.forecast import QuantileForecaster, evaluate
from solarsense.monitoring.anomaly import anomaly_mask, detect
from solarsense.monitoring.drift import drift_report, psi


def test_forecast_beats_persistence_and_quantiles_are_ordered(df, perf):
    mask = anomaly_mask(df.index, detect(df, perf, 3.0))
    n = 90 * 96
    model = QuantileForecaster(5000.0, 0).fit(df.iloc[:n], mask[:n])
    pred = model.predict(df)
    assert (pred["p10"] <= pred["p50"]).all() and (pred["p50"] <= pred["p90"]).all()
    assert (pred.loc[df["clearsky_ghi"] == 0] == 0).all().all()
    m = evaluate(df.iloc[n:], pred.iloc[n:], 5000.0, mask[n:])
    assert m["nmae_pct"] < m["persistence_nmae_pct"]
    assert 55 < m["coverage_p10_p90_pct"] < 95


def test_dsm_penalty_only_charges_outside_tolerance():
    # capacity 1000 kW, 10 % band = 100 kW; deviation 300 kW -> 200 kW excess * 0.25 h * 2 INR
    p = penalty([700.0, 950.0], [1000.0, 1000.0], 1000.0, tariff=2.0, tolerance_pct=10.0)
    assert np.isclose(p[0], 200 * 0.25 * 2.0)
    assert p[1] == 0.0


def test_psi_detects_distribution_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert psi(a, rng.normal(0, 1, 5000)) < 0.05
    assert psi(a, rng.normal(2, 1, 5000)) > 0.25


def test_drift_report_flags_retrain_on_error_growth():
    idx = pd.date_range("2026-01-01", periods=200, freq="15min")
    d = pd.DataFrame({"clearsky_ghi": 1.0, "ghi_fc": 1.0, "temp_fc": 1.0, "cloud_fc": 1.0}, index=idx)
    assert drift_report(d, d, nmae_ref=3.0, nmae_cur=6.0)["retrain_recommended"]
    assert not drift_report(d, d, nmae_ref=3.0, nmae_cur=3.2)["retrain_recommended"]
