import importlib.util
from pathlib import Path

import pytest

from embodied_agent_harness.agents import checked_proposal, load_agent
from embodied_agent_harness.rsi import PickPlaceSkill

FEEDBACK = {"failure": "empty_grasp", "close_vertical_clearance_m": 0.1, "close_xy_error_m": 0.001}


def test_builtin_agent_and_decline():
    agent = load_agent("height-repair")
    candidate, explanation = checked_proposal(agent, PickPlaceSkill(), FEEDBACK)
    assert candidate.grasp_offset_m == pytest.approx(0.03)
    assert explanation["candidate_skill"] == candidate.identity
    assert checked_proposal(agent, candidate, {**FEEDBACK, "failure": None}) is None


def test_custom_agent_example_changes_strategy():
    path = Path(__file__).resolve().parents[1] / "examples/custom_revision_agent.py"
    spec = importlib.util.spec_from_file_location("custom_agent", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidate, explanation = checked_proposal(module.create_agent(), PickPlaceSkill(), FEEDBACK)
    assert candidate.grasp_offset_m == 0.025
    assert explanation["candidate_skill"] == candidate.identity


def test_reject_unchanged_candidate_and_preserve_feedback():
    class BadAgent:
        def propose(self, skill, feedback):
            feedback["failure"] = "modified"
            return skill, {}

    with pytest.raises(ValueError, match="unchanged"):
        checked_proposal(BadAgent(), PickPlaceSkill(), FEEDBACK)
    assert FEEDBACK["failure"] == "empty_grasp"


def test_factory_loader(monkeypatch):
    class Module:
        @staticmethod
        def create():
            return load_agent("height-repair")

    monkeypatch.setattr("importlib.import_module", lambda name: Module)
    assert callable(load_agent("my_agents:create").propose)
    with pytest.raises(ValueError, match="module:factory"):
        load_agent("unknown")
