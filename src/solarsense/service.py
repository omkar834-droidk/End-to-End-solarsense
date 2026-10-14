"""PlantService: owns data, models and cached predictions (one instance per process)."""
from __future__ import annotations

import logging
import threading

import numpy as np
import pandas as pd

from solarsense.config import Settings, get_settings
from solarsense.data.simulator import STEPS_PER_DAY, simulate
from solarsense.ml.dsm import compare_schedules
from solarsense.ml.forecast import QuantileForecaster, evaluate, evaluate_rows
from solarsense.monitoring.anomaly import PerformanceModel, anomaly_mask, daily_stats, detect
from solarsense.monitoring.drift import drift_report

log = logging.getLogger(__name__)


class PlantService:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.ready = False
        self.model_version = 1
        self._lock = threading.Lock()

    @property
    def model_path(self):
        return self.s.artifact_dir / "forecaster.joblib"

    def ensure_ready(self) -> "PlantService":
        with self._lock:
            if self.ready:
                return self
            s = self.s
            self.df = simulate(s.sim_start, s.sim_days, s.capacity_kw, s.latitude, s.longitude, s.seed)
            self.perf = PerformanceModel.calibrate(self.df, s.capacity_kw)
            self.anomalies = detect(self.df, self.perf, s.tariff_inr_per_kwh)
            self.mask = anomaly_mask(self.df.index, self.anomalies)
            n_train = s.train_days * STEPS_PER_DAY
            if self.model_path.exists():
                self.model = QuantileForecaster.load(self.model_path)
            else:
                self.model = QuantileForecaster(s.capacity_kw, s.seed).fit(self.df.iloc[:n_train], self.mask[:n_train])
                self.model.save(self.model_path)
            self.n_eval_start = n_train
            self._refresh_predictions()
            self.val_rows = (np.arange(len(self.df)) >= n_train) & ~self.mask
            self.train_rows = (np.arange(len(self.df)) < n_train) & ~self.mask
            self.ready = True
            log.info("PlantService ready: %s rows, %d anomalies", len(self.df), len(self.anomalies))
        return self

    def _refresh_predictions(self) -> None:
        self.pred = self.model.predict(self.df)
        n = self.n_eval_start
        self.holdout = evaluate(self.df.iloc[n:], self.pred.iloc[n:], self.s.capacity_kw, self.mask[n:])

    def _recent_healthy(self, days: int) -> np.ndarray:
        """Row mask for the last `days` days' worth of fault-free rows (works while a fault is ongoing)."""
        pos = np.flatnonzero(~self.mask)[-days * STEPS_PER_DAY:]
        rows = np.zeros(len(self.df), dtype=bool)
        rows[pos] = True
        return rows

    # ---- views -------------------------------------------------------
    @property
    def as_of(self) -> pd.Timestamp:
        return self.df.index[-1]

    def window(self, days: int) -> pd.DataFrame:
        return self.df.iloc[-days * STEPS_PER_DAY:]

    def latest_pr(self) -> float:
        n = 7 * STEPS_PER_DAY
        return float(daily_stats(self.df.iloc[-n:], self.perf)["pr"].dropna().iloc[-1])

    def performance(self, days: int) -> list[dict]:
        st = daily_stats(self.window(days), self.perf)
        return [{"date": str(d.date()), "pr": None if np.isnan(v) else round(float(v), 4)} for d, v in st["pr"].items()]

    def forecast_day(self, day: str) -> dict | None:
        try:
            sl = self.df.loc[day]
        except KeyError:
            return None
        if sl.empty:
            return None
        p = self.pred.loc[day]
        return {
            "timestamps": [t.isoformat() for t in sl.index],
            "p10": p["p10"].round(1).tolist(), "p50": p["p50"].round(1).tolist(), "p90": p["p90"].round(1).tolist(),
            "actual": sl["power_kw"].round(1).tolist(),
            "faulty": self.mask[self.df.index.get_indexer(sl.index)].tolist(),
        }

    def dsm(self, days: int) -> dict:
        n = days * STEPS_PER_DAY
        s = self.s
        return compare_schedules(self.df.iloc[-n:], self.pred.iloc[-n:], s.capacity_kw, s.tariff_inr_per_kwh,
                                 s.dsm_tolerance_pct, s.dsm_penalty_multiplier, self.mask[-n:])

    def drift(self) -> dict:
        recent = self._recent_healthy(14)
        cur_err = evaluate_rows(self.df, self.pred, self.s.capacity_kw, recent)["nmae_pct"]
        ref_err = self.holdout["nmae_pct"]
        # forecast inputs are unaffected by plant faults, so PSI uses the true last 14 days
        return drift_report(self.df[self.train_rows], self.df.iloc[-14 * STEPS_PER_DAY:], ref_err, cur_err)

    def retrain(self) -> dict:
        """Refit on all healthy data except the latest 14 healthy days, which become the validation set."""
        with self._lock:
            hold = self._recent_healthy(14)
            self.model = QuantileForecaster(self.s.capacity_kw, self.s.seed).fit(self.df, self.mask | hold)
            self.model.save(self.model_path)
            self.model_version += 1
            self.pred = self.model.predict(self.df)
            self.train_rows, self.val_rows = ~(self.mask | hold), hold
            self.holdout = evaluate_rows(self.df, self.pred, self.s.capacity_kw, hold)
        return {"model_version": self.model_version, "validation": self.holdout, "meta": self.model.meta}

    def snapshot(self) -> dict:
        d = self.drift()
        return {
            "latest_pr": self.latest_pr(),
            "open_anomalies": sum(1 for a in detect(self.window(30), self.perf, self.s.tariff_inr_per_kwh) if a.ongoing),
            "nmae_pct": self.holdout["nmae_pct"],
            "max_psi": d["max_psi"],
            "model_version": self.model_version,
        }
