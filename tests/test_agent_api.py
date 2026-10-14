import pytest
from fastapi.testclient import TestClient

from solarsense.agent.graph import build_agent
from solarsense.api.main import app, service


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_and_plant(client):
    assert client.get("/api/health").json()["ready"] is True
    plant = client.get("/api/plant").json()
    assert plant["capacity_kw"] == 5000 and plant["validation"]["nmae_pct"] > 0


def test_forecast_endpoint(client):
    r = client.get("/api/forecast", params={"day": "2026-03-15"}).json()
    assert len(r["p50"]) == 96 and len(r["actual"]) == 96
    assert client.get("/api/forecast", params={"day": "2030-01-01"}).status_code == 404
    assert client.get("/api/forecast", params={"day": "bad"}).status_code == 422


def test_agent_creates_work_order_for_ongoing_string_fault(client):
    out = client.post("/api/diagnose", json={"days": 30}).json()
    assert out["status"] == "attention"
    assert out["llm_used"] is False
    kinds = [f["kind"] for f in out["findings"]]
    assert "string_fault" in kinds
    wo = out["work_orders"]
    assert wo and wo[0]["priority"] == "P2" and wo[0]["checklist"]
    assert out["trace"][0].startswith("Scan")


def test_agent_reports_healthy_window():
    service.ensure_ready()
    agent = build_agent(service)
    # days 30-60 of the simulation contain no fault
    window_service = service
    original = window_service.window
    window_service.window = lambda days: window_service.df.iloc[45 * 96 : 75 * 96]
    try:
        out = agent.invoke({"days": 30, "trace": []})
    finally:
        window_service.window = original
    assert out["status"] == "healthy" and out["work_orders"] == []


def test_dsm_drift_performance_and_metrics(client):
    d = client.get("/api/dsm", params={"days": 14}).json()
    assert d["model_penalty_inr"] <= d["persistence_penalty_inr"]
    assert client.get("/api/drift").json()["status"] in {"ok", "watch", "retrain"}
    assert len(client.get("/api/performance", params={"days": 30}).json()["daily"]) == 30
    assert "solarsense_performance_ratio_latest" in client.get("/metrics").text


def test_retrain_requires_admin_key(client):
    assert client.post("/api/retrain").status_code == 401
    r = client.post("/api/retrain", headers={"X-Admin-Key": "test-admin-key"})
    assert r.status_code == 200 and r.json()["model_version"] == 2
