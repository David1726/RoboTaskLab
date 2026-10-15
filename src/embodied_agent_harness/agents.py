"""Replace the demo's revision agent without replacing physics or acceptance checks."""

import importlib
from typing import Protocol

from .rsi import PickPlaceSkill, propose


class RevisionAgent(Protocol):
    def propose(self, skill: PickPlaceSkill, feedback: dict) -> tuple[PickPlaceSkill, dict] | None:
        """Return a bounded candidate and explanation, or decline to revise."""
        ...


class HeightRepairAgent:
    def propose(self, skill: PickPlaceSkill, feedback: dict):
        return propose(skill, feedback)


def load_agent(name: str) -> RevisionAgent:
    if name == "height-repair":
        return HeightRepairAgent()
    if ":" not in name:
        raise ValueError("agent must be height-repair or an installed module:factory")
    module, factory = name.split(":", 1)
    agent = getattr(importlib.import_module(module), factory)()
    if not callable(getattr(agent, "propose", None)):
        raise ValueError("agent must implement propose(skill, feedback)")
    return agent


def checked_proposal(agent: RevisionAgent, skill: PickPlaceSkill, feedback: dict):
    # The agent receives development feedback only; defensive copies preserve evidence.
    result = agent.propose(skill, dict(feedback))
    if result is None:
        return None
    candidate, explanation = result
    if not isinstance(candidate, PickPlaceSkill) or not isinstance(explanation, dict):
        raise ValueError("revision agent must return PickPlaceSkill and a JSON explanation")
    if candidate.identity == skill.identity:
        raise ValueError("revision agent returned an unchanged skill")
    return candidate, {
        **explanation,
        "parent_skill": skill.identity,
        "candidate_skill": candidate.identity,
    }
