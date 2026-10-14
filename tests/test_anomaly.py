import pandas as pd

from solarsense.data.simulator import simulate
from solarsense.monitoring.anomaly import PerformanceModel, anomaly_mask, detect


def _overlaps(a, truth_start, truth_end, tol="2D"):
    tol = pd.Timedelta(tol)
    return a.start <= truth_end + tol and a.end >= truth_start - tol


def test_detects_every_injected_fault(df, perf):
    found = detect(df, perf, 3.0)
    outages = [a for a in found if a.kind == "inverter_outage"]
    strings = [a for a in found if a.kind == "string_fault"]
    assert len(outages) == 2
    assert len(strings) == 1 and strings[0].ongoing
    # each detected outage lines up with an injected one
    start = df.index[0]
    for a, (d0, d1) in zip(outages, [(40.3, 40.7), (112.4, 113.2)], strict=True):
        assert _overlaps(a, start + pd.Timedelta(days=d0), start + pd.Timedelta(days=d1))
    assert 18 < strings[0].severity_pct < 28  # injected 22 %


def test_no_false_fault_alarms_on_healthy_plant():
    df = simulate(faults=())
    perf = PerformanceModel.calibrate(df, 5000.0)
    kinds = {a.kind for a in detect(df, perf, 3.0)}
    assert not kinds & {"inverter_outage", "string_fault"}


def test_soiling_is_flagged_when_nothing_cleans_the_panels():
    df = simulate(faults=(), cleaning_days=(0,))
    perf = PerformanceModel.calibrate(df, 5000.0)
    assert "soiling" in {a.kind for a in detect(df, perf, 3.0)}


def test_mask_covers_fault_rows_only(df, perf):
    mask = anomaly_mask(df.index, detect(df, perf, 3.0))
    truth = (df["fault_label"] != "none").values
    assert mask[truth].mean() > 0.9   # catches the faulty rows
    assert mask[~truth].mean() < 0.15  # without masking much healthy data
