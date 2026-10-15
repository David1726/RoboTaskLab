"""Export actual transfer evidence; redact host paths without modifying raw runs."""

import argparse
import hashlib
import shutil
from pathlib import Path

from embodied_agent_harness.portable import portable_json
from embodied_agent_harness.report import render_benchmark_report, render_transfer_report
from embodied_agent_harness.spec import read_json, write_json
from embodied_agent_harness.transfer import paired_comparison


def export(run: Path, source_run: Path, out: Path):
    project = Path(__file__).resolve().parents[1]
    run, source_run, out = run.resolve(), source_run.resolve(), out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    provenance = []

    def copy(source, destination):
        if not any(
            source.resolve().is_relative_to(root) for root in [run, source_run]
        ) or not destination.resolve().is_relative_to(out):
            raise ValueError("evidence export path outside run or output")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".json":
            write_json(
                destination, portable_json(read_json(source, legacy_nonfinite=True), project)
            )
        else:
            shutil.copy2(source, destination)
        provenance.append(
            {
                "file": destination.relative_to(out).as_posix(),
                "original_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "export_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            }
        )

    report = read_json(run / "report.json", legacy_nonfinite=True)
    for name in ["manifest.json", "experience-bank.json"]:
        copy(run / name, out / name)
    rows = {}
    for condition in report["conditions"]:
        copy(run / condition / "summary.json", out / condition / "summary.json")
        rows[condition] = read_json(out / condition / "summary.json")["rows"]
        for row in rows[condition]:
            case = row["case"]["id"]
            for name in [
                "result.json",
                "context.json",
                "trajectory.jsonl",
                "rollout.mp4",
                "initial-state.npy",
                "latest.png",
            ]:
                copy(run / condition / case / name, out / condition / case / name)
            if (run / condition / case / "physical-state.json").is_file():
                copy(
                    run / condition / case / "physical-state.json",
                    out / condition / case / "physical-state.json",
                )
    source_summary = read_json(source_run / "summary.json", legacy_nonfinite=True)
    source_cases = {r["case"]["id"] for r in source_summary["rows"]}
    bank_cases = {r["case_id"] for r in read_json(run / "experience-bank.json")["experiences"]}
    if source_cases != bank_cases:
        raise ValueError("export source evidence does not match the context bank")
    for name in ["manifest.json", "summary.json"]:
        copy(source_run / name, out / "source" / name)
    for row in source_summary["rows"]:
        case = row["case"]["id"]
        for name in ["result.json", "trajectory.jsonl", "rollout.mp4", "initial-state.npy"]:
            copy(source_run / case / name, out / "source" / case / name)
        if (source_run / case / "physical-state.json").is_file():
            copy(
                source_run / case / "physical-state.json",
                out / "source" / case / "physical-state.json",
            )
    for row in source_summary["rows"]:
        case = row["case"]["id"]
        if (source_run / case / "context.json").is_file():
            copy(source_run / case / "context.json", out / "source" / case / "context.json")
    metrics = paired_comparison(rows)
    for key in ["evaluation_track", "generalization_axis", "physical_icl", "limitation"]:
        if key in report:
            metrics[key] = report[key]
    metrics["source_cost"] = {
        "episodes": len(source_summary["rows"]),
        "steps": sum(r["steps"] for r in source_summary["rows"]),
        "skill_calls": sum(r["skill_calls"] for r in source_summary["rows"]),
    }
    write_json(out / "report.json", metrics)
    render_transfer_report(out)
    render_benchmark_report(out / "source")
    write_json(
        out / "export-manifest.json",
        {
            "schema": "eah.evidence_export.v1",
            "files": provenance,
            "note": "JSON host paths redacted; undefined diagnostics exported as null. "
            "Media/trajectory bytes unchanged. "
            "Raw runs preserved.",
        },
    )
    print(out / "report.html")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    export(args.run, args.source_run, args.out)


if __name__ == "__main__":
    main()
