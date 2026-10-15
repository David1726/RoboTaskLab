"""Paired cross-task context experiments with frozen skill and agent code."""

import argparse
import hashlib
import importlib.metadata
from pathlib import Path

from .benchmark import (
    campaign_state,
    create_case_env,
    execute_case,
    execution_identity,
    load_panel,
    protocol_files,
    summarize,
    verify_frozen,
)
from .memory import CONDITIONS, build_bank, select_context
from .metrics import METRIC_VERSION
from .physical_icl import CONDITIONS as PHYSICAL_CONDITIONS
from .physical_icl import select_physical_context, validate_protocol
from .spec import write_json


def paired_comparison(rows_by_condition):
    baseline = {r["case"]["id"]: r for r in rows_by_condition["no_context"]}
    if not baseline:
        raise ValueError("nonempty baseline required")
    comparisons = {}
    for condition, rows in rows_by_condition.items():
        index = {r["case"]["id"]: r for r in rows}
        if index.keys() != baseline.keys() or len(index) != len(rows):
            raise ValueError("condition has missing/duplicate cases")
        for name, row in index.items():
            if row["case"] != baseline[name]["case"]:
                raise ValueError("task definitions differ across paired conditions")
            if row["initial_state_sha256"] != baseline[name]["initial_state_sha256"]:
                raise ValueError("initial states differ across paired conditions")
            for field in ["init_asset_sha256", "metric_version"]:
                if row.get(field) != baseline[name].get(field):
                    raise ValueError(
                        "environment definitions or metric versions differ across pairs"
                    )
        successes = sum(r["success"] for r in rows)
        successful_rows = [r for r in rows if r["success"]]
        changes = [
            {
                "case_id": key,
                "task_id": row["case"]["task_id"],
                "task_family": row["case"].get("task_family", "basket_pick_place"),
                "world_id": row["case"].get("world_id"),
                "success_delta": int(row["success"]) - int(baseline[key]["success"]),
                "step_delta": row["steps"] - baseline[key]["steps"],
                "skill_call_delta": row["skill_calls"] - baseline[key]["skill_calls"],
            }
            for key, row in index.items()
        ]
        comparisons[condition] = {
            "episodes": len(rows),
            "successes": successes,
            "success_rate": successes / len(rows),
            "paired_success_delta": sum(r["success_delta"] for r in changes) / len(rows),
            "mean_steps": sum(r["steps"] for r in rows) / len(rows),
            "mean_skill_calls": sum(r["skill_calls"] for r in rows) / len(rows),
            "implementation_errors": sum(
                r.get("termination") == "implementation_error" for r in rows
            ),
            "mean_steps_on_success": (
                sum(r["steps"] for r in successful_rows) / len(successful_rows)
                if successful_rows
                else None
            ),
            "paired_successful_cases": sum(
                row["success"] and baseline[key]["success"] for key, row in index.items()
            ),
            "mean_step_delta_on_paired_success": (
                sum(
                    row["steps"] - baseline[key]["steps"]
                    for key, row in index.items()
                    if row["success"] and baseline[key]["success"]
                )
                / sum(row["success"] and baseline[key]["success"] for key, row in index.items())
                if any(row["success"] and baseline[key]["success"] for key, row in index.items())
                else None
            ),
            "negative_transfer_cases": sum(p["success_delta"] < 0 for p in changes),
            "positive_transfer_cases": sum(p["success_delta"] > 0 for p in changes),
            "pairs": changes,
        }
    return {
        "schema": "eah.transfer_report.v1",
        "conditions": comparisons,
        "limitation": "Pilot task families and finite samples. Rule-agent context transfer is "
        "an engineering demonstration, not evidence of neural physical ICL.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--panel", type=Path, default=Path("benchmarks/transfer-targets.json"))
    parser.add_argument("--skills", type=Path, default=Path("participant/skills.py"))
    parser.add_argument("--agent", type=Path, default=Path("participant/agent.py"))
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
    )
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    panel = load_panel(args.panel)
    physical = panel.get("evaluation_track") == "physical_icl"
    supported = PHYSICAL_CONDITIONS if physical else CONDITIONS
    if args.conditions is None:
        args.conditions = (
            [
                "no_context",
                "source_context",
                "action_effect",
                "language_only",
                "unrelated_physics",
                "shuffled_feedback",
            ]
            if physical
            else ["no_context", "source_context", "failure_only", "shuffled_feedback"]
        )
    if args.shots is None:
        args.shots = 1 if physical else 2
    if (
        "no_context" not in args.conditions
        or len(args.conditions) != len(set(args.conditions))
        or any(c not in supported for c in args.conditions)
        or args.shots < (1 if physical else 0)
    ):
        parser.error(
            "unique supported conditions including no_context and nonnegative shots required"
        )
    cases = [c for c in panel["cases"] if c["split"] == "evaluation"]
    if not cases:
        parser.error("nonempty evaluation target cases required")
    bank = build_bank(args.source_run.resolve(), args.skills.resolve())
    if physical:
        validate_protocol(
            panel,
            bank,
            expected_metric_version=METRIC_VERSION,
            expected_execution_identity=execution_identity(),
            expected_agent_sha256=hashlib.sha256(args.agent.read_bytes()).hexdigest(),
        )
    if set(bank["source_tasks"]) & {c["task_id"] for c in cases}:
        parser.error("source and target task identities overlap")
    root = args.out.resolve()
    hashes = {
        str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [args.skills, args.agent, args.panel]
    }
    frozen = {
        **hashes,
        **{str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protocol_files()},
    }
    for condition in args.conditions:
        for case in cases:
            if physical:
                select_physical_context(bank, case, condition, args.shots)
            else:
                select_context(bank, case["task_id"], condition, args.shots)
    if args.source is None and any(c.get("backend", "libero") == "libero" for c in cases):
        parser.error("LIBERO cases require --source")
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "experience-bank.json", bank)
    write_json(
        root / "manifest.json",
        {
            "panel": panel,
            "conditions": args.conditions,
            "shots": args.shots,
            "source_bank_id": bank["id"],
            "participant_sha256": hashes,
            "versions": {
                p: importlib.metadata.version(p) for p in ["mujoco", "robosuite", "numpy"]
            },
            "evaluation_scope": "physical_icl_cross_family"
            if physical
            else "task_identity_transfer",
            "evaluation_track": panel.get("evaluation_track", "agent"),
            "metric_version": METRIC_VERSION if physical else None,
            "execution_identity": execution_identity(),
        },
    )
    with campaign_state(root):
        rows_by_condition = {}
        for condition in args.conditions:
            condition_root = root / condition
            condition_root.mkdir()
            rows_by_condition[condition] = []
            for case in cases:
                verify_frozen(frozen)
                context = (
                    select_physical_context(bank, case, condition, args.shots)
                    if physical
                    else select_context(bank, case["task_id"], condition, args.shots)
                )
                directory = condition_root / case["id"]
                directory.mkdir()
                write_json(directory / "context.json", context)
                env, task, states, init_hash = create_case_env(args.source, directory, case)
                try:
                    row = execute_case(
                        env, task, states, case, panel, args.skills, args.agent, directory, context
                    )
                    verify_frozen(frozen)
                    row["init_asset_sha256"] = init_hash
                    write_json(directory / "result.json", row)
                    rows_by_condition[condition].append(row)
                    write_json(
                        condition_root / "summary.json", summarize(rows_by_condition[condition])
                    )
                    print(
                        f"{condition}/{case['id']}: success={row['success']}, steps={row['steps']}",
                        flush=True,
                    )
                finally:
                    env.close()
        report = paired_comparison(rows_by_condition)
        report["evaluation_track"] = panel.get("evaluation_track", "agent")
        report["generalization_axis"] = "task_family" if physical else "task_identity"
        if physical:
            from .physical_icl import physical_summary

            report["physical_icl"] = physical_summary(rows_by_condition)
        report["limitation"] = (
            "Frozen rule/system-identification baseline; no neural ICL claim. "
            "State observations, limited manufacturing tasks and finite physical worlds."
        )
        report["source_cost"] = {
            "episodes": len(bank["experiences"]),
            "steps": sum(e["steps"] for e in bank["experiences"]),
            "skill_calls": sum(e["skill_calls"] for e in bank["experiences"]),
        }
        write_json(root / "report.json", report)
        from .report import render_transfer_report

        render_transfer_report(root)
        print(root / "report.html")
        return int(
            any(
                r["termination"] == "implementation_error"
                for rows in rows_by_condition.values()
                for r in rows
            )
        )


if __name__ == "__main__":
    raise SystemExit(main())
