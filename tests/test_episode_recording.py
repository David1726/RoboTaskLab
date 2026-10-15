"""Recorder contract tests with fixture state/cameras; no physics or model inference."""

from types import SimpleNamespace

import pytest

from embodied_agent_harness.benchmark import execute_case
from embodied_agent_harness.spec import read_json, strict_json

np = pytest.importorskip("numpy")
pytest.importorskip("imageio")
pytest.importorskip("PIL")


class FixtureEnv:
    def __init__(self, invalid_observation=False):
        self.env = self
        self.case = {"backend": "manufacturing"}
        self.invalid = invalid_observation
        self.objects_dict = {"peg": None}
        self.obj_body_id = {"peg": 0}
        self.sim = SimpleNamespace(data=SimpleNamespace(body_xpos=np.zeros((1, 3))))
        self.obs = {
            "agentview_image": np.zeros((384, 512, 3), dtype=np.uint8),
            "robot0_eef_pos": np.array([0.0, 0.0, 1.0]),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.zeros(2),
        }

    def initialize(self, case):
        return self.obs

    def get_sim_state(self):
        return np.zeros(3)

    def check_success(self):
        return False

    def step(self, action):
        if self.invalid:
            self.obs["robot0_eef_pos"][0] = np.nan
        return self.obs, 0, False, {}


@pytest.mark.parametrize("failure", ["factory", "import", "feedback", "observation"])
def test_failures_keep_complete_episode_records_and_clean_actor_frames(
    tmp_path, monkeypatch, failure
):
    monkeypatch.chdir(tmp_path)
    agent = tmp_path / "agent.py"
    skills = tmp_path / "skills.py"
    agent.write_text(
        "class Agent:\n def decide(self, observation, feedback):\n"
        '  return {"type":"skill","name":"insert","arguments":{}}\n'
        "def create_agent(): return Agent()\n"
    )
    if failure == "factory":
        agent.write_text('def create_agent(): raise RuntimeError("factory failure")\n')
    elif failure == "import":
        agent.write_text("this is invalid Python syntax !")
    skills.write_text(
        "def execute(call, context):\n context.step([0]*7)\n"
        ' return {"status":"completed","value":float("nan")}\n'
    )
    output = tmp_path / "episode"
    output.mkdir()
    row = execute_case(
        FixtureEnv(failure == "observation"),
        SimpleNamespace(language="insert"),
        None,
        {"id": "case-0", "backend": "manufacturing", "task_id": "insert_round"},
        {"max_steps": 10, "max_skill_calls": 1},
        skills,
        agent,
        output,
    )
    assert row["termination"] == "implementation_error" and not row["success"]
    assert row == read_json(output / "result.json")
    assert (output / "latest.png").is_file() and (output / "rollout.mp4").stat().st_size > 0
    trace = [strict_json(line) for line in (output / "trajectory.jsonl").read_text().splitlines()]
    assert len(trace) == row["steps"] == (0 if failure in {"factory", "import"} else 1)
    if row["calls"]:
        assert row["calls"][0]["end_step"] == row["steps"]
    if failure == "observation":
        assert trace[0]["observation"] is None
    assert not list((tmp_path / ".cache" / "actor-frames").glob("*.png"))
