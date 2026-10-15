"""Source-only experience banks with explicit provenance and target-leakage checks."""

import hashlib
import json
from pathlib import Path

from .spec import read_json

CONDITIONS = {"no_context", "source_context", "failure_only", "shuffled_feedback"}


def content_id(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def build_bank(source_run: Path, skills: Path) -> dict:
    manifest = read_json(source_run / "manifest.json")
    source_summary = read_json(source_run / "summary.json")
    if manifest.get("split") != "development":
        raise ValueError("source campaign must contain only development episodes")
    if manifest.get("status") not in {None, "completed"}:
        raise ValueError("source campaign did not complete")
    skills_hash = hashlib.sha256(skills.read_bytes()).hexdigest()
    recorded = manifest.get("entrypoint_sha256", {}).get("skills")
    if recorded is None:
        # Legacy records use host paths; portable exports still identify the skill by name/hash.
        matches = [
            h
            for p, h in manifest["participant_sha256"].items()
            if p.replace("\\", "/").rsplit("/", 1)[-1] == skills.name
        ]
        recorded = matches[0] if len(matches) == 1 else None
    if recorded != skills_hash:
        raise ValueError(
            "source experience was not collected with the current frozen skill implementation"
        )
    rows = source_summary["rows"]
    expected = [
        c
        for c in manifest["panel"]["cases"]
        if manifest["split"] == "all" or c["split"] == manifest["split"]
    ]
    if len(rows) != len(expected) or {r["case"]["id"] for r in rows} != {c["id"] for c in expected}:
        raise ValueError("source campaign must be complete")
    experiences = []
    expected_cases = {c["id"]: c for c in expected}
    for row in rows:
        if row["case"] != expected_cases[row["case"]["id"]]:
            raise ValueError("source case differs from the frozen panel")
        if row.get("evidence_kind") != "simulator_execution" or row["termination"] not in {
            "agent_stop",
            "step_budget",
            "skill_call_budget",
        }:
            raise ValueError("source bank requires actual execution without unresolved code errors")
        directory = (source_run / row["case"]["id"]).resolve()
        if not directory.is_relative_to(source_run.resolve()):
            raise ValueError("source episode path outside campaign")
        journal = directory / "trajectory.jsonl"
        video = directory / "rollout.mp4"
        result = read_json(directory / "result.json")
        if result != row or hashlib.sha256(video.read_bytes()).hexdigest() != row["video_sha256"]:
            raise ValueError("source summary/media differs from episode evidence")
        trajectory_hash = hashlib.sha256(journal.read_bytes()).hexdigest()
        if row.get("trajectory_sha256") and row["trajectory_sha256"] != trajectory_hash:
            raise ValueError("source trajectory differs from recorded hash")
        initial_file = directory / "initial-state.npy"
        if row.get("initial_state_file_sha256") and (
            hashlib.sha256(initial_file.read_bytes()).hexdigest()
            != row["initial_state_file_sha256"]
        ):
            raise ValueError("source initial state differs from recorded hash")
        trajectory = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
        if len(trajectory) != row["steps"]:
            raise ValueError("source trajectory does not match execution step count")
        # Bounded state/action excerpt; no repeated nested context or absolute local paths.
        indices = sorted({round(i * (len(trajectory) - 1) / 15) for i in range(16)})
        excerpt = [
            {
                "step": trajectory[i]["step"],
                "action": trajectory[i]["action"],
                "eef_pos": trajectory[i]["observation"]["eef_pos"],
                "objects": trajectory[i]["observation"]["objects"],
                "wrist_force_N": trajectory[i]["observation"].get("wrist_force_N"),
                "wrist_torque_Nm": trajectory[i]["observation"].get("wrist_torque_Nm"),
            }
            for i in indices
            if trajectory
        ]
        experience = {
            "metric_version": row.get("metric_version", "legacy_v1"),
            "task_id": row["case"]["task_id"],
            "task_family": row["case"].get("task_family", "basket_pick_place"),
            "world_id": row["case"].get("world_id"),
            "source_layout": {
                "backend": row["case"].get("backend", "libero"),
                "seed": row["case"].get("layout_seed", row["case"].get("init_state_index")),
                "opening_m": row["case"].get("opening_m", 0.014)
                if str(row["case"]["task_id"]).startswith("insert_")
                else None,
            },
            "physics_sha256": content_id(row["case"].get("physics")),
            "instruction": row["instruction"],
            "case_id": row["case"]["id"],
            "steps": row["steps"],
            "skill_calls": len(row["calls"]),
            "success": row["success"],
            "peak_wrist_force_N": row.get("peak_wrist_force_N"),
            "calls": row["calls"],
            "initial_state_sha256": row["initial_state_sha256"],
            "trajectory_sha256": trajectory_hash,
            "trajectory_excerpt": excerpt,
            "integrity_coverage": (
                "recorded_media_trace_state"
                if row.get("trajectory_sha256") and row.get("initial_state_file_sha256")
                else "legacy_media_recorded_trace_imported"
            ),
            "video_sha256": row["video_sha256"],
        }
        experience["id"] = content_id(experience)
        experiences.append(experience)
    bank = {
        "schema": "eah.experience_bank.v1",
        "metric_versions": sorted({e["metric_version"] for e in experiences}),
        "execution_identity": manifest.get("execution_identity"),
        "agent_sha256": manifest.get("entrypoint_sha256", {}).get("agent"),
        "skill_sha256": skills_hash,
        "source_tasks": sorted({e["task_id"] for e in experiences}, key=str),
        "experiences": experiences,
    }
    bank["id"] = content_id(bank)
    return bank


def select_context(bank: dict, target_task: int, condition: str, shots: int) -> dict:
    if condition not in CONDITIONS or shots < 0:
        raise ValueError("invalid condition/context shot count")
    if target_task in bank["source_tasks"]:
        raise ValueError("cross-task target appears in source experience bank")
    if condition == "no_context":
        selected = []
    else:
        if len(bank["experiences"]) < shots:
            raise ValueError("insufficient source context shots")
        selected = json.loads(json.dumps(bank["experiences"][:shots]))
    if condition == "failure_only":
        for experience in selected:
            experience["calls"] = [
                c for c in experience["calls"] if c.get("feedback", {}).get("status") != "completed"
            ]
            experience["success"] = False
    if condition == "shuffled_feedback":
        # Corrupted-context control only. Original simulator journals and bank remain unchanged.
        for experience in selected:
            feedbacks = [c.get("feedback") for c in experience["calls"]]
            feedbacks = feedbacks[1:] + feedbacks[:1]
            for call, feedback in zip(experience["calls"], feedbacks, strict=True):
                call["feedback"] = feedback
    return {
        "mode": condition,
        "bank_id": bank["id"],
        "shots": len(selected),
        "experiences": selected,
        "transformation": condition,
    }
