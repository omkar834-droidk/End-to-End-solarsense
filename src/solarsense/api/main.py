import logging
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from solarsense.agent.graph import build_agent
from solarsense.monitoring.anomaly import detect
from solarsense.service import PlantService

log = logging.getLogger("solarsense.api")
WEB = Path(__file__).resolve().parents[3] / "web"

service = PlantService()
LATENCY = Histogram("solarsense_request_seconds", "Request latency", ["path"])
G_PR = Gauge("solarsense_performance_ratio_latest", "Latest daily performance ratio")
G_OPEN = Gauge("solarsense_open_anomalies", "Ongoing anomalies in the last 30 days")
G_NMAE = Gauge("solarsense_forecast_nmae_pct", "Forecast nMAE on the validation window (%)")
G_PSI = Gauge("solarsense_drift_psi_max", "Max PSI across watched features")
G_VER = Gauge("solarsense_model_version", "Serving model version")


@asynccontextmanager
async def lifespan(_: FastAPI):
    service.ensure_ready()
    app.state.agent = build_agent(service)
    yield


app = FastAPI(title="SolarSense", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def timing(request: Request, call_next):
    t = time.perf_counter()
    response = await call_next(request)
    if request.url.path.startswith("/api"):
        LATENCY.labels(request.url.path).observe(time.perf_counter() - t)
    return response


class DiagnoseRequest(BaseModel):
    days: int = Field(30, ge=7, le=150)


@app.get("/api/health")
def health():
    return {"status": "ok", "ready": service.ready, "as_of": str(service.as_of) if service.ready else None}


@app.get("/api/plant")
def plant():
    s = service.s
    return {"name": s.plant_name, "capacity_kw": s.capacity_kw, "as_of": str(service.as_of),
            "model_version": service.model_version, "model": service.model.meta,
            "validation": service.holdout, "perf_model": {"k": round(service.perf.k, 4), "ref_pr": round(service.perf.ref_pr, 4)}}


@app.get("/api/forecast")
def forecast(day: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$")):
    data = service.forecast_day(day or str(service.as_of.date()))
    if data is None:
        raise HTTPException(404, "No data for that day")
    return data


@app.get("/api/anomalies")
def anomalies(days: int = Query(30, ge=7, le=150)):
    found = detect(service.window(days), service.perf, service.s.tariff_inr_per_kwh)
    return {"days": days, "anomalies": [a.to_dict() for a in found]}


@app.get("/api/performance")
def performance(days: int = Query(60, ge=7, le=150)):
    return {"days": days, "reference": round(service.perf.ref_pr, 4), "daily": service.performance(days)}


@app.get("/api/dsm")
def dsm(days: int = Query(14, ge=2, le=60)):
    return service.dsm(days)


@app.get("/api/drift")
def drift():
    return service.drift()


@app.post("/api/diagnose")
def diagnose(body: DiagnoseRequest):
    try:
        return app.state.agent.invoke({"days": body.days, "trace": []})
    except Exception:
        log.exception("diagnose failed")
        raise HTTPException(500, "Diagnosis failed. Please try again.") from None


@app.post("/api/retrain")
def retrain(x_admin_key: str = Header(default="")):
    key = service.s.admin_api_key
    if not key:
        raise HTTPException(503, "Retraining is disabled: set SOLARSENSE_ADMIN_API_KEY")
    if not secrets.compare_digest(x_admin_key.encode(), key.encode()):
        raise HTTPException(401, "Invalid admin key")
    return service.retrain()


@app.get("/metrics")
def metrics():
    snap = service.snapshot()
    G_PR.set(snap["latest_pr"])
    G_OPEN.set(snap["open_anomalies"])
    G_NMAE.set(snap["nmae_pct"])
    G_PSI.set(snap["max_psi"])
    G_VER.set(snap["model_version"])
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB / "index.html")


app.mount("/static", StaticFiles(directory=str(WEB)), name="static")
