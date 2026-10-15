"""Use EAH_AGENT_COMMAND (a JSON argv array) to supply your inference worker."""

import json
import os
import sys
from pathlib import Path

from embodied_agent_harness.bridges import JsonProcessAgent


def create_agent():
    default = [sys.executable, str(Path(__file__).with_name("json_worker_demo.py"))]
    return JsonProcessAgent(json.loads(os.environ.get("EAH_AGENT_COMMAND", json.dumps(default))))
