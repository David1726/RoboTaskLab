from copy import deepcopy

import pytest

from embodied_agent_harness.memory import build_bank, content_id, select_context
from embodied_agent_harness.report import clustered_interval, render_transfer_report
from embodied_agent_harness.spec import write_json
from embodied_agent_harness.transfer import paired_comparison


def bank():
    return {
        "id": "source-bank",
        "source_tasks": [0],
        "experiences": [
            {
                "id": "e0",
                "success": True,
                "task_id": 0,
                "calls": [
                    {
                        "decision": {"arguments": {"grasp_offset_m": 0.12}},
                        "feedback": {"status": "empty_grasp"},
                    },
                    {
                        "decision": {"arguments": {"grasp_offset_m": 0.03}},
                        "feedback": {"status": "completed"},
                    },
                ],
            }
        ],
    }


def test_context_never_contains_target_experience():
    with pytest.raises(ValueError, match="target"):
        select_context(bank(), 0, "source_context", 1)
    assert not select_context(bank(), 1, "no_context", 1)["experiences"]
    assert select_context(bank(), 1, "source_context", 1)["experiences"][0]["task_id"] == 0


def test_corruption_control_does_not_modify_ground_truth_bank():
    original = bank()
    snapshot = deepcopy(original)
    context = select_context(original, 1, "shuffled_feedback", 1)
    assert context["experiences"][0]["calls"][0]["feedback"]["status"] == "completed"
    assert original == snapshot
    failures = select_context(original, 1, "failure_only", 1)
    assert len(failures["experiences"][0]["calls"]) == 1
    assert not failures["experiences"][0]["success"]


def rows(prefix, steps=100, success=True):
    return [
        {
            "case": {"id": prefix, "task_id": 1},
            "initial_state_sha256": "initial",
            "success": success,
            "steps": steps,
            "skill_calls": 1,
        }
    ]


def test_paired_efficiency_and_changed_initial_state():
    data = {"no_context": rows("task-1", 200), "source_context": rows("task-1", 100)}
    report = paired_comparison(data)
    assert report["conditions"]["source_context"]["pairs"][0]["step_delta"] == -100
    data["source_context"][0]["initial_state_sha256"] = "changed"
    with pytest.raises(ValueError, match="initial"):
        paired_comparison(data)


def test_failed_memory_transfer_is_not_hidden():
    report = paired_comparison(
        {"no_context": rows("task", success=True), "source_context": rows("task", success=False)}
    )
    assert report["conditions"]["source_context"]["paired_success_delta"] == -1


def test_content_identity_and_clustered_interval_are_deterministic():
    assert content_id({"a": 1, "b": 2}) == content_id({"b": 2, "a": 1})
    pairs = [{"task_id": 1, "success_delta": 0}, {"task_id": 2, "success_delta": 1}]
    assert clustered_interval(pairs) == clustered_interval(pairs)
    assert clustered_interval(pairs)["task_clusters"] == 2


def source_fixture(tmp_path):
    """Synthetic file-integrity fixture only; this test does not simulate a robot."""
    import hashlib
    import json

    source = tmp_path / "source"
    episode = source / "case-0"
    episode.mkdir(parents=True)
    skills = tmp_path / "skills.py"
    skills.write_text("# fixture")
    video, trace, initial = (
        episode / "rollout.mp4",
        episode / "trajectory.jsonl",
        episode / "initial-state.npy",
    )
    video.write_bytes(b"synthetic unit fixture media")
    trace.write_text(
        json.dumps(
            {"step": 1, "action": [0] * 7, "observation": {"eef_pos": [0, 0, 0], "objects": {}}}
        )
        + "\n"
    )
    initial.write_bytes(b"synthetic state")
    case = {"id": "case-0", "task_id": 0, "split": "development"}
    row = {
        "case": case,
        "steps": 1,
        "success": False,
        "calls": [],
        "instruction": "source",
        "termination": "agent_stop",
        "evidence_kind": "simulator_execution",
        "initial_state_sha256": "state",
        "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "trajectory_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
        "initial_state_file_sha256": hashlib.sha256(initial.read_bytes()).hexdigest(),
    }
    write_json(
        source / "manifest.json",
        {
            "panel": {"cases": [case]},
            "split": "development",
            "participant_sha256": {
                str(skills.resolve()): hashlib.sha256(skills.read_bytes()).hexdigest()
            },
        },
    )
    write_json(source / "summary.json", {"rows": [row]})
    write_json(episode / "result.json", row)
    return source, skills, episode


def test_bank_keeps_failed_source_and_trace_excerpt(tmp_path):
    source, skills, _ = source_fixture(tmp_path)
    result = build_bank(source, skills)
    assert not result["experiences"][0]["success"]
    assert len(result["experiences"][0]["trajectory_excerpt"]) == 1
    assert result["experiences"][0]["integrity_coverage"] == "recorded_media_trace_state"


@pytest.mark.parametrize("file", ["rollout.mp4", "trajectory.jsonl", "initial-state.npy"])
def test_bank_rejects_modified_evidence(tmp_path, file):
    source, skills, episode = source_fixture(tmp_path)
    (episode / file).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash|evidence"):
        build_bank(source, skills)


def test_bank_rejects_changed_skills(tmp_path):
    source, skills, _ = source_fixture(tmp_path)
    skills.write_text("# changed")
    with pytest.raises(ValueError, match="frozen skill"):
        build_bank(source, skills)


def test_source_bank_survives_relocation_with_same_skill_content(tmp_path):
    source, skills, _ = source_fixture(tmp_path)
    relocated = tmp_path / "another-host" / skills.name
    relocated.parent.mkdir()
    relocated.write_bytes(skills.read_bytes())
    assert build_bank(source, relocated)["source_tasks"] == [0]


@pytest.mark.parametrize(
    "field,value", [("split", "evaluation"), ("split", "all"), ("status", "failed")]
)
def test_source_bank_rejects_evaluation_or_failed_campaign(tmp_path, field, value):
    from embodied_agent_harness.spec import read_json

    source, skills, _ = source_fixture(tmp_path)
    manifest = read_json(source / "manifest.json")
    manifest[field] = value
    write_json(source / "manifest.json", manifest)
    with pytest.raises(ValueError, match="development|complete"):
        build_bank(source, skills)


def test_report_recomputes_scores_instead_of_trusting_stale_aggregates(tmp_path):
    from embodied_agent_harness.spec import read_json

    row = {
        "case": {"id": "t0", "task_id": 0},
        "initial_state_sha256": "s",
        "success": False,
        "steps": 20,
        "skill_calls": 1,
        "instruction": "fixture",
        "termination": "agent_stop",
        "error": None,
    }
    write_json(tmp_path / "no_context" / "summary.json", {"rows": [row]})
    write_json(
        tmp_path / "report.json",
        {"evaluation_track": "agent", "conditions": {"no_context": {"successes": 999}}},
    )
    render_transfer_report(tmp_path)
    result = read_json(tmp_path / "report.json")
    assert result["conditions"]["no_context"]["successes"] == 0
    assert result["conditions"]["no_context"]["paired_success_delta_ci95"]["lower"] is None


def test_html_escapes_injected_language_and_links_videos(tmp_path):
    row = rows("case-1")[0]
    row.update(
        {
            "instruction": "</script><script>alert(1)</script>",
            "termination": "agent_stop",
            "error": None,
        }
    )
    write_json(tmp_path / "report.json", paired_comparison({"no_context": [row]}))
    directory = tmp_path / "no_context"
    directory.mkdir()
    write_json(directory / "summary.json", {"rows": [row]})
    content = render_transfer_report(tmp_path).read_text(encoding="utf-8")
    assert row["instruction"] not in content
    assert "no_context/case-1/rollout.mp4" in content
    assert "\\u003c/script>" in content
