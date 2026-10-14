"""O&M knowledge base the agent uses to turn a detected anomaly into an actionable work order."""

PLAYBOOK = {
    "inverter_outage": {
        "asset": "Inverter / AC block",
        "sla_hours": 2,
        "crew": "Electrical O&M (on-call)",
        "causes": [
            "Inverter trip (grid fault, over-temperature, DC isolator open)",
            "AC breaker or transformer protection operated",
            "Communication loss masking a running inverter (check SCADA vs meter)",
        ],
        "checklist": [
            "Read the inverter alarm/event log and note the fault code and trip time",
            "Compare SCADA output with the revenue meter to rule out a data gap",
            "Inspect AC breaker, transformer protection relay and DC isolators",
            "Reset only after the cause is understood; record the restart time",
        ],
    },
    "string_fault": {
        "asset": "DC strings / combiner box",
        "sla_hours": 48,
        "crew": "DC / module O&M team",
        "causes": [
            "Blown string fuse or loose MC4 connector",
            "Bypass diode failure or damaged module",
            "New shading (vegetation, structure) on part of the array",
        ],
        "checklist": [
            "Compare string currents at the combiner box to find the weak string(s)",
            "Run thermography for hot spots on modules and connectors",
            "Check fuses, connectors and bypass diodes on the weak strings",
            "Look for new shading at the affected time of day",
        ],
    },
    "soiling": {
        "asset": "Array (module surface)",
        "sla_hours": 336,
        "crew": "Module cleaning crew",
        "causes": ["Dust and bird-dropping accumulation with no recent rain or cleaning"],
        "checklist": [
            "Confirm with a visual inspection and a reference-cell comparison",
            "Schedule cleaning for the dirtiest blocks first",
            "Re-check performance ratio 2 days after cleaning",
        ],
    },
}
