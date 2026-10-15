"""A second local rule agent. No LLM, VLA, provider credentials, or network calls."""

from dataclasses import replace

from embodied_agent_harness.rsi import propose


class ContactMarginAgent:
    def propose(self, skill, feedback):
        hypothesis = propose(skill, feedback)
        if hypothesis is None:
            return None
        candidate, explanation = hypothesis
        # Independently test a smaller grasp-height margin through the same gate.
        return replace(candidate, grasp_offset_m=min(candidate.grasp_offset_m, 0.025)), {
            **explanation,
            "method": "custom rule agent with a 25 mm grasp offset cap",
        }


def create_agent():
    return ContactMarginAgent()
