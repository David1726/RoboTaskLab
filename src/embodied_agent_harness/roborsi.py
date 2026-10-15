"""Shell-free adapter for the audited RoboRSI eval-suite CLI."""

import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .spec import write_json


def jobs(spec: dict, python: Path, source: Path, out: Path) -> list[dict]:
    result = []
    for panel, settings in spec["panels"].items():
        for seed in settings["seeds"]:
            campaign = out / panel / f"seed-{seed}"
            command = [
                str(python),
                "-m",
                "roborsi",
                "eval-suite",
                "--backend",
                spec["backend"],
                "--atomic",
                spec["atomic"],
                "--pass-at",
                "1",
                "--seed-start",
                str(seed),
                "--workers",
                "1",
                "--tool-budget",
                str(spec["tool_budget"]),
                "--infra-retries",
                str(spec["infra_retries"]),
                "--reasoning-effort",
                spec["reasoning_effort"],
                "--run-mode",
                "frozen",
                "--agent-mode",
                "roborsi",
                "--json",
                "--out",
                str(campaign),
            ]
            for role, model in spec["models"].items():
                command += [f"--{role}-model", model]
            for task in settings["tasks"]:
                command += ["--task", task]
            result.append(
                {
                    "panel": panel,
                    "seed": seed,
                    "cwd": str(source),
                    "command": command,
                    "campaign": str(campaign),
                    "audit_command": [
                        str(python),
                        "-m",
                        "roborsi",
                        "eval-audit",
                        str(campaign),
                        "--check-media",
                        "--require-complete",
                        "--json",
                    ],
                }
            )
    return result


def check_source(source: Path, revision: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full Git SHA")
    for args, expected in (
        (["rev-parse", "HEAD"], revision),
        (["status", "--porcelain", "--untracked-files=all"], ""),
    ):
        proc = subprocess.run(["git", *args], cwd=source, text=True, capture_output=True)
        if proc.returncode or proc.stdout.strip() != expected:
            raise ValueError("RoboRSI checkout must be clean and match the pinned revision")
    if not (source / "roborsi" / "__main__.py").is_file():
        raise ValueError("source is not a RoboRSI checkout")


def execute(
    spec: dict, python: Path, source: Path, revision: str, out: Path, home: Path, timeout: int
) -> dict:
    """Execute only when CLI --execute was explicitly supplied; no automatic retries."""
    check_source(source, revision)
    if timeout < 1:
        raise ValueError("timeout must be positive")
    if any("REPLACE_" in model for model in spec["models"].values()):
        raise ValueError("replace model placeholders before execution")
    # Each run is fresh. No mixing old candidate trajectories or implicit resumes.
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(source),
            "ROBORSI_LIBRARY_ROOT": str(source),
            "ROBORSI_HOME": str(home),
            "ROBORSI_RUN_MODE": "frozen",
            "ROBORSI_ATOMIC_COMPOUND": "1",
            "PYTHONIOENCODING": "utf-8",
        }
    )
    receipt = {
        "schema": "eah.run.v1",
        "evidence_kind": "simulator_execution",
        "revision": revision,
        "spec": spec,
        "source": str(source),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "jobs": [],
    }
    write_json(out / "run.json", receipt)
    for job in jobs(spec, python, source, out):
        check_source(source, revision)
        log = out / f"{job['panel']}-seed-{job['seed']}.log"
        record = {**job, "log": str(log), "status": "started"}
        receipt["jobs"].append(record)
        write_json(out / "run.json", receipt)
        with log.open("w", encoding="utf-8") as stream:
            try:
                proc = subprocess.run(
                    job["command"],
                    cwd=source,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                )
                record.update(status="finished", returncode=proc.returncode)
                if proc.returncode == 0:
                    audit = subprocess.run(
                        job["audit_command"],
                        cwd=source,
                        env=env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        timeout=timeout,
                    )
                    record["audit_returncode"] = audit.returncode
                check_source(source, revision)
            except subprocess.TimeoutExpired:
                record["status"] = "timeout"
            except (OSError, ValueError) as exc:
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(out / "run.json", receipt)
        if (
            record.get("status") != "finished"
            or record.get("returncode") != 0
            or record.get("audit_returncode") != 0
        ):
            break
    return receipt
