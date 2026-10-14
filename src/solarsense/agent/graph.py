"""LangGraph O&M agent: scan -> (healthy | quantify -> diagnose -> prioritize -> work orders -> report).

Tools are deterministic and auditable. An LLM (optional) only phrases the final report from
verified facts; with no API key a template is used, so the agent works offline.
"""
from __future__ import annotations

import json
import logging
from typing import Literal, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from solarsense.agent.playbook import PLAYBOOK
from solarsense.monitoring.anomaly import daily_stats, detect

log = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    days: int
    anomalies: list[dict]
    latest_pr: float
    findings: list[dict]
    work_orders: list[dict]
    report: str
    status: str
    llm_used: bool
    trace: list[str]


def _trace(state: AgentState, msg: str) -> list[str]:
    return [*state.get("trace", []), msg]


def build_agent(service):
    s = service.s

    def scan(state: AgentState):
        window = service.window(state["days"])
        found = [a.to_dict() for a in detect(window, service.perf, s.tariff_inr_per_kwh)]
        pr = daily_stats(window, service.perf)["pr"].dropna()
        return {"anomalies": found, "latest_pr": float(pr.iloc[-1]) if len(pr) else float("nan"),
                "trace": _trace(state, f"Scan -> {len(found)} anomalies in last {state['days']} days")}

    def route(state: AgentState) -> Literal["quantify", "healthy"]:
        return "quantify" if state["anomalies"] else "healthy"

    def healthy(state: AgentState):
        last_pr = state["latest_pr"]
        report = (f"No faults detected in the last {state['days']} days. "
                  f"Latest daily performance ratio is {last_pr:.3f} (reference {service.perf.ref_pr:.3f}).")
        return {"status": "healthy", "findings": [], "work_orders": [], "report": report, "llm_used": False,
                "trace": _trace(state, "Healthy -> no work orders")}

    def quantify(state: AgentState):
        out = []
        for a in state["anomalies"]:
            span_days = max((pd.Timestamp(a["end"]) - pd.Timestamp(a["start"])).total_seconds() / 86400, 1.0)
            out.append({**a, "loss_inr_per_day": round(a["lost_inr"] / span_days)})
        return {"findings": out, "trace": _trace(state, "Quantify -> energy and revenue loss per finding")}

    def diagnose(state: AgentState):
        out = []
        for f in state["findings"]:
            pb = PLAYBOOK[f["kind"]]
            out.append({**f, "asset": pb["asset"], "probable_causes": pb["causes"]})
        return {"findings": out, "trace": _trace(state, "Diagnose -> mapped to O&M playbook")}

    def prioritize(state: AgentState):
        out = []
        for f in state["findings"]:
            if f["kind"] == "inverter_outage":
                pri = "P1" if f["ongoing"] else "P3"
            elif f["kind"] == "string_fault":
                pri = "P2" if f["ongoing"] else "P4"
            else:  # soiling: clean now only if two weeks of loss outweigh the cleaning cost
                pri = "P3" if 14 * f["loss_inr_per_day"] > s.cleaning_cost_inr else "P4"
            out.append({**f, "priority": pri})
        out.sort(key=lambda f: (f["priority"], -f["lost_inr"]))
        status = "critical" if any(f["priority"] == "P1" for f in out) else "attention"
        return {"findings": out, "status": status, "trace": _trace(state, f"Prioritize -> status {status.upper()}")}

    def work_orders(state: AgentState):
        orders = []
        for n, f in enumerate(state["findings"], 1):
            pb = PLAYBOOK[f["kind"]]
            if f["kind"] != "soiling" and not f["ongoing"]:
                continue  # resolved events are reported but need no action
            orders.append({
                "id": f"WO-{pd.Timestamp(f['start']).strftime('%Y%m%d')}-{n:02d}",
                "priority": f["priority"],
                "title": f"{f['kind'].replace('_', ' ').title()} on {pb['asset']}",
                "sla_hours": pb["sla_hours"] if f["priority"] != "P3" or f["kind"] == "soiling" else 24,
                "crew": pb["crew"],
                "est_loss_inr_per_day": f["loss_inr_per_day"],
                "checklist": pb["checklist"],
            })
        return {"work_orders": orders, "trace": _trace(state, f"Work orders -> {len(orders)} created")}

    def narrate(state: AgentState):
        facts = {"status": state["status"], "window_days": state["days"],
                 "findings": [{k: f[k] for k in ("kind", "start", "end", "severity_pct", "lost_kwh", "lost_inr",
                                                  "ongoing", "priority", "probable_causes")} for f in state["findings"]],
                 "work_orders": [{k: w[k] for k in ("id", "priority", "title", "sla_hours")} for w in state["work_orders"]]}
        text = _llm_report(s, facts)
        used = text is not None
        return {"report": text or _template_report(facts), "llm_used": used,
                "trace": _trace(state, f"Report -> {'LLM' if used else 'template'}")}

    g = StateGraph(AgentState)
    for name, fn in [("scan", scan), ("healthy", healthy), ("quantify", quantify), ("diagnose", diagnose),
                     ("prioritize", prioritize), ("work_orders", work_orders), ("narrate", narrate)]:
        g.add_node(name, fn)
    g.add_edge(START, "scan")
    g.add_conditional_edges("scan", route, {"quantify": "quantify", "healthy": "healthy"})
    g.add_edge("quantify", "diagnose")
    g.add_edge("diagnose", "prioritize")
    g.add_edge("prioritize", "work_orders")
    g.add_edge("work_orders", "narrate")
    g.add_edge("narrate", END)
    g.add_edge("healthy", END)
    return g.compile()


def _template_report(facts: dict) -> str:
    lines = [f"Plant status: {facts['status'].upper()} (last {facts['window_days']} days)."]
    for f in facts["findings"]:
        state = "ONGOING" if f["ongoing"] else "resolved"
        lines.append(f"- [{f['priority']}] {f['kind'].replace('_', ' ')} ({state}): {f['severity_pct']}% loss, "
                     f"{f['lost_kwh']:,} kWh / INR {f['lost_inr']:,} since {f['start'][:10]}.")
    for w in facts["work_orders"]:
        lines.append(f"Action {w['id']}: {w['title']} - respond within {w['sla_hours']} h.")
    return "\n".join(lines)


def _llm_report(settings, facts: dict) -> str | None:
    if not settings.groq_api_key:
        return None
    try:
        from langchain_groq import ChatGroq

        llm = ChatGroq(model=settings.groq_model, temperature=0, groq_api_key=settings.groq_api_key)
        prompt = ("You are a solar O&M engineer writing a short shift-handover note. Use ONLY the facts in the JSON. "
                  "Do not invent numbers, causes or actions. Max 120 words.\n" + json.dumps(facts))
        return str(llm.invoke(prompt).content).strip()
    except Exception:  # offline, bad key or rate limit: fall back to the template
        log.exception("LLM report failed; using template")
        return None
