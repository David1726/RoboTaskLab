import math
from pathlib import Path

import pytest

from embodied_agent_harness.benchmark import (
    BudgetExceeded,
    load_module,
    load_panel,
    summarize,
    validate_action,
)
from embodied_agent_harness.spec import write_json

ROOT = Path(__file__).resolve().parents[1]


def test_panel_split_overlap_and_safe_ids(tmp_path):
    panel = load_panel(ROOT / "benchmarks/libero-object-pilot.json")
    path = tmp_path / "panel.json"
    panel["cases"][1]["init_state_index"] = 0
    write_json(path, panel)
    with pytest.raises(ValueError, match="overlap"):
        load_panel(path)
    panel["cases"][1]["init_state_index"] = 1
    panel["cases"][0]["id"] = "../escape"
    write_json(path, panel)
    with pytest.raises(ValueError, match="safe"):
        load_panel(path)


@pytest.mark.parametrize("action", [[0] * 6, [0] * 6 + [float("nan")], [0] * 6 + [2], [True] * 7])
def test_invalid_actions_are_not_executed(action):
    with pytest.raises(ValueError):
        validate_action(action)


class FixtureContext:
    """Contract-only fixture: does not simulate a robot or supply success labels."""

    def __init__(self, maximum=5):
        self.maximum, self.steps = maximum, 0

    def observe(self):
        return {
            "schema": "eah.observation.v1",
            "remaining_steps": self.maximum - self.steps,
            "eef_pos": [0, 0, 0.3],
            "objects": {"object": [0, 0, 0.04], "basket": [0, 0.2, 0.04]},
            "bindings": {"target": "object", "destination": "basket"},
        }

    def step(self, action):
        validate_action(action)
        if self.steps >= self.maximum:
            raise BudgetExceeded()
        self.steps += 1
        return self.observe()


def test_participant_skill_respects_context_budget():
    module = load_module(ROOT / "participant/skills.py", "test_skills")
    context = FixtureContext()
    with pytest.raises(BudgetExceeded):
        module.execute(
            {"name": "pick_place", "arguments": {"target": "object", "destination": "basket"}},
            context,
        )
    assert context.steps == 5


def test_agent_recovers_from_feedback_without_success_oracle():
    module = load_module(ROOT / "participant/agent.py", "test_agent")
    agent = module.create_agent()
    observation = {
        "remaining_steps": 550,
        "instruction": "pick the alphabet soup and place it in the basket",
        "objects": {"alphabet_soup_1": [0, 0, 0], "basket_1": [0, 0, 0]},
    }
    first = agent.decide(observation, None)
    assert first["arguments"]["grasp_offset_m"] == 0.12
    second = agent.decide(observation, {"status": "empty_grasp", "close_vertical_clearance_m": 0.1})
    assert math.isclose(second["arguments"]["grasp_offset_m"], 0.03)
    assert agent.decide(observation, {"status": "completed"}) == {"type": "stop"}


def test_summary_reports_final_success_separately_from_skill_claims():
    rows = [
        {
            "case": {"split": "evaluation"},
            "success": False,
            "termination": "agent_stop",
            "calls": [{"feedback": {"status": "completed"}}],
        },
        {"case": {"split": "development"}, "success": True, "termination": "agent_stop"},
    ]
    report = summarize(rows)
    assert report["success_rate"] == 0.5
    assert report["by_split"]["evaluation"]["success_rate"] == 0
