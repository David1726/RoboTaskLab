"""EDIT BLOCK 2: choose skills, respond to feedback, and decide when to stop."""

from embodied_agent_harness.language import target_for_instruction


class Agent:
    def __init__(self):
        self.calls = 0
        # Deliberate initial defect demonstrates feedback-driven parameter recovery.
        self.offset = 0.12

    def decide(self, observation, feedback):
        if "task_family" in observation:
            return self.manufacturing_decision(observation, feedback)
        if self.calls == 0:
            # Input-context adaptation only; no code or low-level skill changes.
            for experience in observation.get("context", {}).get("experiences", []):
                if not experience.get("success"):
                    continue
                for call in experience["calls"]:
                    if call.get("feedback", {}).get("status") == "completed":
                        offset = call["decision"]["arguments"].get("grasp_offset_m")
                        if isinstance(offset, (float, int)) and -0.01 <= offset <= 0.15:
                            self.offset = offset
        if feedback and feedback["status"] == "completed":
            return {"type": "stop"}
        if feedback and feedback["status"] == "empty_grasp":
            self.offset = max(
                -0.005, min(0.03, self.offset - feedback["close_vertical_clearance_m"] + 0.01)
            )
        if self.calls >= 2 or observation["remaining_steps"] < 250:
            return {"type": "stop"}
        self.calls += 1
        target, destination = target_for_instruction(
            observation["instruction"], observation["objects"]
        )
        return {
            "type": "skill",
            "name": "pick_place",
            "arguments": {
                "grasp_offset_m": self.offset,
                "target": target,
                "destination": destination,
            },
        }

    def manufacturing_decision(self, observation, feedback):
        if feedback and feedback["status"] == "completed":
            return {"type": "stop"}
        if self.calls == 0 and observation["episode_role"] == "source":
            self.calls += 1
            return {"type": "skill", "name": "probe_cartesian", "arguments": {}}
        scale = 1.0
        # Analytic system-identification baseline. It is not an LLM/VLA ICL result.
        for experience in observation.get("context", {}).get("experiences", []):
            for call in experience.get("calls", []):
                if call.get("decision", {}).get("name") == "probe_cartesian":
                    gain = call.get("feedback", {}).get("response_gain")
                    if isinstance(gain, (float, int)) and 0.02 < gain < 3:
                        # Nominal servo response reference; freeze before target evaluation.
                        scale = max(0.4, min(2.5, 0.24 / gain))
        if self.calls >= 3 or observation["remaining_steps"] < 300:
            return {"type": "stop"}
        family = observation["task_family"]
        name = {"insertion": "insert", "nut_assembly": "assemble_nut", "stacking": "stack"}[family]
        self.calls += 1
        return {
            "type": "skill",
            "name": name,
            "arguments": {
                "motion_scale": scale,
                "grasp_offset_m": 0.0 if family == "nut_assembly" else 0.015,
            },
        }


def create_agent():
    return Agent()
