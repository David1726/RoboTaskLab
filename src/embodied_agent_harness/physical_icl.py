"""Physical ICL protocol, distinct from general agent task-success evaluation."""

import json
import random

from .memory import content_id
from .metrics import force_summary

CONDITIONS = {
    "no_context",
    "source_context",
    "action_effect",
    "language_only",
    "unrelated_physics",
    "shuffled_feedback",
}


def validate_protocol(
    panel,
    bank,
    expected_metric_version=None,
    expected_execution_identity=None,
    expected_agent_sha256=None,
):
    if panel.get("evaluation_track") != "physical_icl":
        raise ValueError("physical_icl track declaration required")
    if expected_metric_version is not None and bank.get("metric_versions") != [
        expected_metric_version
    ]:
        raise ValueError("source metric version differs; collect a fresh source campaign")
    if (
        expected_execution_identity is not None
        and bank.get("execution_identity") != expected_execution_identity
    ):
        raise ValueError("source environment/runtime differs; collect a fresh source campaign")
    if expected_agent_sha256 is not None and bank.get("agent_sha256") != expected_agent_sha256:
        raise ValueError("source agent differs from the frozen target agent")
    targets = [c for c in panel["cases"] if c["split"] == "evaluation"]
    if not targets or not bank["experiences"]:
        raise ValueError("nonempty source and target cases required")
    source_families = {e["task_family"] for e in bank["experiences"]}
    target_families = {c["task_family"] for c in targets}
    if source_families & target_families:
        raise ValueError("physical ICL cross-family source and target families overlap")
    if set(bank["source_tasks"]) & {c["task_id"] for c in targets}:
        raise ValueError("source and target task identities overlap")
    layouts = {}
    for experience in bank["experiences"]:
        if "source_layout" not in experience:
            raise ValueError("source layout provenance required; rebuild bank from source run")
        layouts.setdefault(experience["world_id"], set()).add(
            (experience["task_id"], content_id(experience["source_layout"]))
        )
    if any(v != next(iter(layouts.values())) for v in layouts.values()):
        raise ValueError(
            "counterfactual source tasks/layouts/geometry differ across physical worlds"
        )
    for case in targets:
        matches = [e for e in bank["experiences"] if e["world_id"] == case["world_id"]]
        if not matches or any(e["physics_sha256"] != content_id(case["physics"]) for e in matches):
            raise ValueError(
                "target world lacks source experience with matching physical conditions"
            )


def select_physical_context(bank, case, condition, shots):
    if condition not in CONDITIONS or shots < 1:
        raise ValueError("supported physical context condition and positive shots required")
    if condition == "no_context":
        selected = []
    else:
        unrelated = condition == "unrelated_physics"
        candidates = [
            e
            for e in bank["experiences"]
            if (
                e["world_id"] != case["world_id"]
                if unrelated
                else e["world_id"] == case["world_id"]
            )
            and (
                e["physics_sha256"] != content_id(case["physics"])
                if unrelated
                else e["physics_sha256"] == content_id(case["physics"])
            )
        ]
        if len(candidates) < shots:
            raise ValueError(
                "insufficient matched or unrelated source shots; never duplicate samples"
            )

        def signature(experience):
            return (str(experience["task_id"]), content_id(experience["source_layout"]))

        matched = sorted(
            [e for e in bank["experiences"] if e["world_id"] == case["world_id"]],
            key=signature,
        )[:shots]
        if len(matched) < shots:
            raise ValueError("insufficient matched source shots")
        if unrelated:
            selected = []
            for reference in matched:
                matches = sorted(
                    [e for e in candidates if signature(e) == signature(reference)],
                    key=lambda e: (e["world_id"], e.get("case_id", "")),
                )
                if not matches:
                    raise ValueError(
                        "unrelated-physics control must keep the source task and layout"
                    )
                selected.append(matches[0])
        else:
            selected = matched
        selected = json.loads(json.dumps(selected))
    if condition == "language_only":
        selected = [
            {"instruction": e["instruction"], "task_family": e["task_family"]} for e in selected
        ]
    elif condition == "action_effect":
        selected = [
            {
                "calls": [c for c in e["calls"] if c["decision"]["name"] == "probe_cartesian"],
                "trajectory_excerpt": e["trajectory_excerpt"],
            }
            for e in selected
        ]
    elif condition == "shuffled_feedback":
        for experience in selected:
            feedbacks = [c.get("feedback") for c in experience["calls"]]
            for call, feedback in zip(
                experience["calls"], feedbacks[1:] + feedbacks[:1], strict=True
            ):
                call["feedback"] = feedback
    # Do not reveal the experimental condition to the participant as a shortcut.
    return {"mode": "physical_context", "shots": len(selected), "experiences": selected}


def physical_summary(rows_by_condition):
    """Equal family weight inside each world; bootstrap the physical-world unit."""
    if not rows_by_condition.get("no_context"):
        raise ValueError("nonempty no_context baseline required")
    reference = rows_by_condition["no_context"]
    expected = {(r["case"]["world_id"], r["case"]["task_family"]) for r in reference}
    for rows in rows_by_condition.values():
        coverage = {(r["case"]["world_id"], r["case"]["task_family"]) for r in rows}
        if coverage != expected:
            raise ValueError("physical world/family coverage differs across conditions")
    grouped = {}
    for condition, rows in rows_by_condition.items():
        worlds = {}
        for row in rows:
            case = row["case"]
            worlds.setdefault(case["world_id"], {}).setdefault(case["task_family"], []).append(
                int(row["success"])
            )
        grouped[condition] = {
            w: {f: sum(v) / len(v) for f, v in families.items()} for w, families in worlds.items()
        }
    baseline = grouped["no_context"]
    result = {}
    for condition, worlds in grouped.items():
        deltas = [
            sum(value - baseline[w][f] for f, value in families.items()) / len(families)
            for w, families in worlds.items()
        ]
        score = sum(sum(f.values()) / len(f) for f in worlds.values()) / len(worlds)
        rng = random.Random(0)
        samples = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(2000))
        interval_status = (
            "insufficient_worlds"
            if len(worlds) < 5
            else "degenerate_resampling"
            if samples[50] == samples[1950]
            else "exploratory"
        )
        result[condition] = {
            "world_family_macro_success": score,
            "by_world_family": worlds,
            "paired_macro_delta": sum(deltas) / len(deltas),
            "paired_macro_delta_ci95": {
                "estimate": sum(deltas) / len(deltas),
                "lower": samples[50] if interval_status == "exploratory" else None,
                "upper": samples[1950] if interval_status == "exploratory" else None,
                "status": interval_status,
                "world_clusters": len(worlds),
                "method": "equal family weight within world; world-cluster bootstrap",
                "warning": "Suppress intervals with fewer than five worlds or degenerate "
                "resampling. This display guard is not a statistical sample-size guarantee. "
                "Any displayed interval remains exploratory, not a significance test.",
            },
        }
        rows = rows_by_condition[condition]
        result[condition].update(force_summary(rows))
    return result
