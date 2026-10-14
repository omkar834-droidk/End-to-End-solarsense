import numpy as np

from solarsense.data.simulator import simulate


def test_shape_and_physical_limits(df):
    assert len(df) == 150 * 96
    assert df["power_kw"].between(0, 5000).all()
    assert df.loc[df["clearsky_ghi"] == 0, "power_kw"].max() == 0
    assert df["ghi"].max() < 1200


def test_deterministic_for_same_seed():
    a, b = simulate(days=5, seed=7), simulate(days=5, seed=7)
    assert np.allclose(a["power_kw"], b["power_kw"])


def test_injected_faults_are_labelled(df):
    assert set(df["fault_label"]) == {"none", "inverter_outage", "string_fault"}
