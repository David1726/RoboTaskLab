import json
import math
from pathlib import Path

from embodied_agent_harness.portable import portable_json


def test_portable_export_handles_nested_paths_and_nonfinite_diagnostics():
    project = Path.cwd().resolve()
    raw = {
        str(project / "participant" / "agent.py"): [
            {"value": -math.inf, "file": str(project / "runs" / "case" / "result.json")},
            math.nan,
        ]
    }
    exported = portable_json(raw, project)
    parsed = json.loads(json.dumps(exported, allow_nan=False))
    assert parsed["<project>/participant/agent.py"][0] == {
        "value": None,
        "file": "<project>/runs/case/result.json",
    }
    assert parsed["<project>/participant/agent.py"][1] is None
    assert raw[str(project / "participant" / "agent.py")][0]["value"] == -math.inf
