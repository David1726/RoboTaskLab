import subprocess
import sys
from pathlib import Path

import pytest

from embodied_agent_harness.bridges import JsonProcessAgent


def test_process_roundtrip_preserves_state_and_feedback():
    worker = (
        "import sys,json; r=json.load(sys.stdin); "
        "json.dump({'schema':'eah.agent_response.v1',"
        "'decision':{'type':'stop'},'state':{'n':(r['state'] or {}).get('n',0)+1,"
        "'feedback':r['feedback']}},sys.stdout)"
    )
    agent = JsonProcessAgent([sys.executable, "-c", worker])
    assert agent.decide({"instruction": "goal"}, None) == {"type": "stop"}
    agent.decide({}, {"status": "empty_grasp"})
    assert agent.state == {"n": 2, "feedback": {"status": "empty_grasp"}}


@pytest.mark.parametrize(
    "worker,exception",
    [
        ("print('{}')", ValueError),
        ("raise SystemExit(2)", RuntimeError),
        ("print('not json')", ValueError),
    ],
)
def test_bad_worker_output_is_an_error_not_a_success(worker, exception):
    with pytest.raises(exception):
        JsonProcessAgent([sys.executable, "-c", worker]).decide({}, None)


def test_default_worker_runs_without_simulator_or_paid_service(monkeypatch):
    from embodied_agent_harness.benchmark import load_module

    monkeypatch.delenv("EAH_AGENT_COMMAND", raising=False)
    module = load_module(Path("examples/process_agent.py"), "process_test")
    agent = module.create_agent()
    observation = {
        "instruction": "把字母汤放进篮子",
        "objects": {"alphabet_soup_1": [0, 0, 0], "basket_1": [1, 1, 1]},
        "remaining_steps": 600,
    }
    decision = agent.decide(observation, None)
    assert decision["arguments"]["target"] == "alphabet_soup_1"
    retry = agent.decide(observation, {"status": "empty_grasp", "close_vertical_clearance_m": 0.1})
    assert retry["arguments"]["grasp_offset_m"] < decision["arguments"]["grasp_offset_m"]
    assert agent.decide(observation, {"status": "completed"}) == {"type": "stop"}


def test_worker_timeout_is_bounded():
    worker = JsonProcessAgent([sys.executable, "-c", "import time; time.sleep(5)"], 0.1)
    with pytest.raises(subprocess.TimeoutExpired):
        worker.decide({}, None)
