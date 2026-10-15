"""Zero-cost protocol example. Replace Agent.decide with actual model inference.

Reads one JSON request from stdin and writes one JSON response to stdout.
Diagnostics belong on stderr. This worker is a rule agent, not an LLM.
"""

import json
import sys
from pathlib import Path

from embodied_agent_harness.benchmark import load_module


def main():
    request = json.load(sys.stdin)
    if request.get("schema") != "eah.agent_request.v1":
        raise ValueError("unsupported request schema")
    module = load_module(
        Path(__file__).resolve().parents[1] / "participant/agent.py", "demo_worker_agent"
    )
    agent = module.create_agent()
    if request.get("state") is not None:
        agent.calls = request["state"]["calls"]
        agent.offset = request["state"]["offset"]
    decision = agent.decide(request["observation"], request["feedback"])
    json.dump(
        {
            "schema": "eah.agent_response.v1",
            "decision": decision,
            "state": {"calls": agent.calls, "offset": agent.offset},
        },
        sys.stdout,
    )


if __name__ == "__main__":
    main()
