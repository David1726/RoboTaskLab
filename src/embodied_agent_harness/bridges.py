"""Shell-free JSON process bridge for local models or existing agent runtimes."""

import json
import math
import os
import subprocess

from .spec import strict_json


class JsonProcessAgent:
    """One process request per decision; explicit JSON state survives between requests.

    The executable owns inference. This class has no provider, credentials, model
    download, or simulator access. It is a trusted runtime adapter, not a sandbox.
    """

    def __init__(self, command, timeout_seconds=60):
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(part, str) or not part for part in command)
            or type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("nonempty argument list and positive timeout required")
        self.command = command[:]
        self.timeout_seconds = timeout_seconds
        self.state = None

    def decide(self, observation, feedback):
        request = {
            "schema": "eah.agent_request.v1",
            "observation": observation,
            "feedback": feedback,
            "state": self.state,
        }
        result = subprocess.run(
            self.command,
            input=json.dumps(request, ensure_ascii=False, allow_nan=False),
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=self.timeout_seconds,
            shell=False,
            check=False,
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
        )
        if result.returncode:
            raise RuntimeError(f"agent worker exited with code {result.returncode}")
        response = strict_json(result.stdout)
        if not isinstance(response, dict) or response.get("schema") != "eah.agent_response.v1":
            raise ValueError("worker must emit an eah.agent_response.v1 JSON object")
        decision = response.get("decision")
        if not isinstance(decision, dict) or decision.get("type") not in {"skill", "stop"}:
            raise ValueError("worker decision must be skill or stop")
        if decision["type"] == "skill" and (
            not isinstance(decision.get("name"), str)
            or not decision["name"]
            or not isinstance(decision.get("arguments"), dict)
        ):
            raise ValueError("worker skill decision requires a name and arguments")
        self.state = response.get("state")
        return decision
