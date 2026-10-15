import hashlib
import json
import re
from pathlib import Path

from .spec import expected_campaign, read_json


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(root: Path, spec: dict) -> dict:
    receipt = read_json(root / "run.json")
    kind = receipt.get("evidence_kind")
    if (
        receipt.get("schema") != "eah.run.v1"
        or receipt.get("spec") != spec
        or kind not in {"fixture", "simulator_execution"}
        or not re.fullmatch(r"[0-9a-f]{40}", str(receipt.get("revision", "")))
    ):
        raise ValueError("invalid run receipt or experiment mismatch")
    errors, warnings, files = [], [], {}
    pairs = {}
    runtime = None
    run_ids = set()
    for panel, settings in spec["panels"].items():
        for seed in settings["seeds"]:
            campaign_root = root / panel / f"seed-{seed}"
            manifest_path = campaign_root / "campaign.json"
            journal_path = campaign_root / "episodes.jsonl"
            if not manifest_path.is_file() or not journal_path.is_file():
                errors.append(f"missing campaign/journal: {panel}/seed-{seed}")
                continue
            manifest = read_json(manifest_path)
            files[str(manifest_path)] = digest(manifest_path)
            files[str(journal_path)] = digest(journal_path)
            mismatch = [
                k for k, v in expected_campaign(spec, panel, seed).items() if manifest.get(k) != v
            ]
            if mismatch:
                raise ValueError(f"campaign configuration drift: {mismatch}")
            current_runtime = manifest.get("runtime")
            if (
                not isinstance(current_runtime, dict)
                or current_runtime.get("roborsi_commit") != receipt["revision"]
                or current_runtime.get("roborsi_dirty") is not False
            ):
                raise ValueError("missing or inconsistent source fingerprint")
            if runtime is not None and runtime != current_runtime:
                raise ValueError("runtime drift within a run")
            runtime = current_runtime
            if kind == "simulator_execution":
                matching = [
                    j
                    for j in receipt.get("jobs", [])
                    if j.get("panel") == panel and j.get("seed") == seed
                ]
                if (
                    len(matching) != 1
                    or matching[0].get("status") != "finished"
                    or matching[0].get("returncode") != 0
                    or matching[0].get("audit_returncode") != 0
                ):
                    errors.append(f"execution/audit did not finish: {panel}/seed-{seed}")
                audit_path = campaign_root / "audit.json"
                if audit_path.is_file():
                    audit = read_json(audit_path)
                    files[str(audit_path)] = digest(audit_path)
                    if (
                        audit.get("integrity_status") != "pass"
                        or audit.get("campaign_status") != "complete"
                    ):
                        errors.append(f"upstream audit failed: {panel}/seed-{seed}")
                else:
                    errors.append(f"missing upstream audit: {panel}/seed-{seed}")
            attempts = {}
            terminal = set()
            for line_number, line in enumerate(
                journal_path.read_text(encoding="utf-8").splitlines(), 1
            ):
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("journal rows must be objects")
                task = row.get("task_key")
                if task not in settings["tasks"] or type(row.get("seed")) is not int:
                    raise ValueError("journal task/seed outside the experiment")
                if row["seed"] != seed:
                    raise ValueError("journal seed outside campaign")
                if (
                    row.get("backend") != spec["backend"]
                    or row.get("run_mode") != "frozen"
                    or row.get("agent_mode") != "roborsi"
                ):
                    raise ValueError("journal backend/mode drift")
                key = (panel, task, seed)
                attempt = row.get("attempt")
                if type(attempt) is not int or attempt <= attempts.get(key, 0) or key in terminal:
                    raise ValueError("duplicate, unordered, or post-terminal attempt")
                attempts[key] = attempt
                verdict = row.get("verdict")
                if verdict not in {"success", "failure", "infra", "implementation_error"}:
                    raise ValueError("unknown journal verdict")
                if verdict in {"success", "failure"}:
                    if (
                        row.get("success") is not (verdict == "success")
                        or row.get("status") != "terminal"
                    ):
                        raise ValueError("inconsistent simulator verdict")
                    terminal.add(key)
                    run_id = row.get("run_id")
                    if not isinstance(run_id, str) or not run_id or run_id in run_ids:
                        raise ValueError("missing or reused execution run_id")
                    run_ids.add(run_id)
                elif row.get("success") is not None:
                    raise ValueError("infrastructure/code errors cannot supply success labels")
                video = row.get("video_path")
                workspace = row.get("workspace")
                if kind == "simulator_execution":
                    if video:
                        if not Path(video).is_absolute() or not Path(video).is_file():
                            errors.append(f"missing referenced video: {video}")
                    else:
                        warnings.append(f"no video for {key}")
                    if verdict in {"success", "failure"} and (
                        not workspace
                        or not Path(workspace).is_absolute()
                        or not Path(workspace).is_dir()
                    ):
                        errors.append(f"missing execution workspace: {key}")
                pairs[key] = {
                    "verdict": verdict,
                    "run_id": row.get("run_id"),
                    "journal": str(journal_path),
                    "line": line_number,
                    "video_path": video,
                    "workspace": workspace,
                    "reviewer_verdict": row.get("reviewer_verdict"),
                    "error": row.get("error"),
                    "attempt": attempt,
                }
    for panel, settings in spec["panels"].items():
        for task in settings["tasks"]:
            for seed in settings["seeds"]:
                key = (panel, task, seed)
                if key not in pairs or pairs[key]["verdict"] not in {"success", "failure"}:
                    errors.append(f"no terminal simulator verdict: {key}")
    files[str(root / "run.json")] = digest(root / "run.json")
    return {
        "kind": kind,
        "revision": receipt["revision"],
        "pairs": pairs,
        "runtime": runtime,
        "errors": errors,
        "warnings": warnings,
        "files": files,
    }


def failure_packet(run: dict) -> dict:
    return {
        "schema": "eah.failure_packet.v1",
        "evidence_kind": run["kind"],
        "revision": run["revision"],
        "errors": run["errors"],
        "instruction": "Inspect live evidence before attributing a cause. Reviewer labels are "
        "hypotheses, not simulator success predicates. Propose a separate committed "
        "revision; never change past verdicts. Do not tune on held-out tasks.",
        "development_failures": [
            {"task": key[1], "seed": key[2], **row}
            for key, row in run["pairs"].items()
            if key[0] == "development" and row["verdict"] != "success"
        ],
        "evidence_sha256": run["files"],
    }


def compare(spec: dict, baseline: dict, candidate: dict, proposal: dict) -> dict:
    if baseline["kind"] != candidate["kind"]:
        raise ValueError("fixture and simulator evidence cannot be mixed")
    if (
        proposal.get("schema") != "eah.proposal.v1"
        or proposal.get("baseline_revision") != baseline["revision"]
        or proposal.get("candidate_revision") != candidate["revision"]
        or candidate["revision"] == baseline["revision"]
    ):
        raise ValueError("proposal must bind a distinct baseline and candidate revision")
    if not all(
        isinstance(proposal.get(k), str) and proposal[k].strip()
        for k in ("diagnosis_hypothesis", "change", "rationale")
    ):
        raise ValueError("proposal requires diagnosis_hypothesis, change, rationale")
    targets = proposal.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("proposal requires development failure targets")
    for target in targets:
        if not isinstance(target, dict) or type(target.get("seed")) is not int:
            raise ValueError("invalid proposal target")
        row = baseline["pairs"].get(("development", target.get("task"), target["seed"]))
        if not row or row["verdict"] != "failure":
            raise ValueError("proposal targets must reference baseline development failures")
    base_runtime, cand_runtime = baseline["runtime"], candidate["runtime"]
    if base_runtime is not None and cand_runtime is not None:

        def context(runtime):
            return {k: v for k, v in runtime.items() if k != "roborsi_commit"}

        if context(base_runtime) != context(cand_runtime):
            raise ValueError("baseline/candidate runtime mismatch outside source revision")
    base_ids = {r["run_id"] for r in baseline["pairs"].values() if r["run_id"]}
    cand_ids = {r["run_id"] for r in candidate["pairs"].values() if r["run_id"]}
    if base_ids & cand_ids:
        raise ValueError("candidate reuses baseline execution IDs; fresh execution required")
    stats = {
        p: {
            "expected": len(v["tasks"]) * len(v["seeds"]),
            "paired": 0,
            "baseline_successes": 0,
            "candidate_successes": 0,
            "improvements": 0,
            "regressions": 0,
        }
        for p, v in spec["panels"].items()
    }
    changes = []
    for key, base_row in baseline["pairs"].items():
        cand_row = candidate["pairs"].get(key)
        if (
            base_row["verdict"] not in {"success", "failure"}
            or not cand_row
            or cand_row["verdict"] not in {"success", "failure"}
        ):
            continue
        bucket = stats[key[0]]
        base_ok, cand_ok = base_row["verdict"] == "success", cand_row["verdict"] == "success"
        bucket["paired"] += 1
        bucket["baseline_successes"] += base_ok
        bucket["candidate_successes"] += cand_ok
        bucket["improvements"] += not base_ok and cand_ok
        bucket["regressions"] += base_ok and not cand_ok
        if base_ok != cand_ok:
            changes.append(
                {
                    "panel": key[0],
                    "task": key[1],
                    "seed": key[2],
                    "change": "improvement" if cand_ok else "regression",
                    "baseline": base_row,
                    "candidate": cand_row,
                }
            )
    reasons = baseline["errors"] + candidate["errors"]
    if reasons:
        decision = "inconclusive"
    else:
        if any(v["regressions"] for v in stats.values()):
            reasons.append("at least one previously successful task/seed regressed")
        if not stats["development"]["improvements"]:
            reasons.append("no development improvement")
        decision = "reject" if reasons else "accept_for_review"
    if baseline["kind"] == "fixture":
        decision = "fixture_" + decision
    return {
        "schema": "eah.comparison.v1",
        "evidence_kind": baseline["kind"],
        "decision": decision,
        "reasons": reasons,
        "panels": stats,
        "changes": changes,
        "warnings": baseline["warnings"] + candidate["warnings"],
        "proposal": proposal,
        "evidence_sha256": {**baseline["files"], **candidate["files"]},
        "limitation": "Small finite-panel gate, not statistical or generalization proof. "
        "Journal provenance is trusted, not cryptographically attested. "
        "Acceptance records evidence only; it never installs a skill.",
    }
