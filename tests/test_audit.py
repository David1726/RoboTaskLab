import hashlib
import importlib.util
import json
import math
import os
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from embodied_agent_harness.benchmark import campaign_state, load_module, load_panel, verify_frozen
from embodied_agent_harness.bridges import JsonProcessAgent
from embodied_agent_harness.physical_icl import select_physical_context, validate_protocol
from embodied_agent_harness.spec import read_json, strict_json, write_json


def test_plugin_loading_uses_current_bytes_even_when_cached_timestamp_and_size_match(tmp_path):
    path = tmp_path / "plugin.py"
    path.write_text("value = 1\n")
    spec = importlib.util.spec_from_file_location("cached_plugin", path)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
    before = path.stat()
    path.write_text("value = 2\n")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert load_module(path, "fresh_plugin").value == 2


@pytest.mark.parametrize("text", ['{"x": NaN}', '{"x": 1e999}', '{"x":1,"x":2}'])
def test_strict_json_rejects_ambiguous_or_nonfinite_values(text):
    with pytest.raises(ValueError):
        strict_json(text)


def test_invalid_json_write_preserves_previous_valid_record(tmp_path):
    path = tmp_path / "result.json"
    write_json(path, {"success": False})
    with pytest.raises(ValueError):
        write_json(path, {"force": math.inf})
    assert read_json(path) == {"success": False}
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("timeout", [math.nan, math.inf, True, 0])
def test_process_timeout_must_be_finite_and_positive(timeout):
    with pytest.raises(ValueError):
        JsonProcessAgent([sys.executable], timeout)


def test_process_rejects_nonfinite_state_before_accepting_it():
    worker = [
        sys.executable,
        "-c",
        'print(\'{"schema":"eah.agent_response.v1","decision":{"type":"stop"},"state":NaN}\')',
    ]
    agent = JsonProcessAgent(worker)
    with pytest.raises(ValueError, match="nonfinite"):
        agent.decide({}, None)
    assert agent.state is None


def test_last_episode_code_change_marks_campaign_failed(tmp_path):
    plugin = tmp_path / "agent.py"
    plugin.write_text("# frozen")
    hashes = {str(plugin): hashlib.sha256(plugin.read_bytes()).hexdigest()}
    write_json(tmp_path / "manifest.json", {"panel": {}})
    with pytest.raises(ValueError, match="frozen"):
        with campaign_state(tmp_path):
            plugin.write_text("# changed during final episode")
            verify_frozen(hashes)
    assert read_json(tmp_path / "manifest.json")["status"] == "failed"


def test_source_environment_and_agent_must_match_target_runtime():
    panel = {"evaluation_track": "physical_icl"}
    bank = {"metric_versions": ["v2"], "execution_identity": {"runtime": "old"}}
    with pytest.raises(ValueError, match="environment/runtime"):
        validate_protocol(panel, bank, expected_execution_identity={"runtime": "new"})
    with pytest.raises(ValueError, match="source agent"):
        validate_protocol(panel, bank, expected_agent_sha256="new_agent")


def test_unrelated_shots_match_selected_task_and_layout_even_if_bank_order_differs():
    from embodied_agent_harness.memory import content_id

    experiences = []
    for world in ["w0", "w1"]:
        for task in ["nut_round", "nut_square"] if world == "w0" else ["nut_square", "nut_round"]:
            experiences.append(
                {
                    "world_id": world,
                    "task_id": task,
                    "source_layout": {"seed": 12},
                    "physics_sha256": content_id({"gain": 1 if world == "w0" else 2}),
                }
            )
    bank = {"experiences": experiences}
    case = {"world_id": "w0", "physics": {"gain": 1}}
    before = deepcopy(bank)
    matched = select_physical_context(bank, case, "source_context", 1)
    unrelated = select_physical_context(bank, case, "unrelated_physics", 1)
    assert matched["experiences"][0]["task_id"] == unrelated["experiences"][0]["task_id"]
    assert bank == before


def test_one_world_cannot_have_two_physical_parameter_sets(tmp_path):
    panel = json.loads(Path("benchmarks/manufacturing-agent-v2.json").read_text())
    worlds = {}
    for case in panel["cases"]:
        if case["world_id"] in worlds:
            case["physics"]["mass_scale"] += 0.01
            break
        worlds[case["world_id"]] = case
    path = tmp_path / "panel.json"
    write_json(path, panel)
    with pytest.raises(ValueError, match="consistent physics"):
        load_panel(path)
