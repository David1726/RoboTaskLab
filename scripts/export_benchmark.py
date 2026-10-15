"""Export the independent agent/source execution track with immutable media."""

import argparse
import hashlib
import shutil
from pathlib import Path

from embodied_agent_harness.portable import portable_json
from embodied_agent_harness.report import render_benchmark_report
from embodied_agent_harness.spec import read_json, write_json


def export(run, out):
    project = Path(__file__).resolve().parents[1]
    run, out = run.resolve(), out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    records = []

    files = [Path("manifest.json"), Path("summary.json")]
    for row in read_json(run / "summary.json", legacy_nonfinite=True)["rows"]:
        files.extend(
            Path(row["case"]["id"]) / name
            for name in [
                "result.json",
                "trajectory.jsonl",
                "rollout.mp4",
                "initial-state.npy",
                "latest.png",
            ]
        )
    for relative in files:
        source, destination = run / relative, out / relative
        if not source.resolve().is_relative_to(run) or not destination.resolve().is_relative_to(
            out
        ):
            raise ValueError("episode export path outside run or output")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".json":
            write_json(
                destination, portable_json(read_json(source, legacy_nonfinite=True), project)
            )
        else:
            shutil.copy2(source, destination)
        records.append(
            {
                "file": relative.as_posix(),
                "original_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "export_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            }
        )
    for row in read_json(out / "summary.json")["rows"]:
        relative = Path(row["case"]["id"]) / "context.json"
        if (run / relative).is_file():
            destination = out / relative
            write_json(destination, portable_json(read_json(run / relative), project))
            records.append(
                {
                    "file": relative.as_posix(),
                    "original_sha256": hashlib.sha256((run / relative).read_bytes()).hexdigest(),
                    "export_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                }
            )
        relative = Path(row["case"]["id"]) / "physical-state.json"
        if (run / relative).is_file():
            destination = out / relative
            write_json(destination, portable_json(read_json(run / relative), project))
            records.append(
                {
                    "file": relative.as_posix(),
                    "original_sha256": hashlib.sha256((run / relative).read_bytes()).hexdigest(),
                    "export_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                }
            )
    write_json(
        out / "export-manifest.json",
        {
            "schema": "eah.evidence_export.v1",
            "files": records,
            "note": "JSON host paths redacted; undefined diagnostics exported as null. "
            "Media/state/trajectory bytes unchanged; "
            "raw runs preserved.",
        },
    )
    print(render_benchmark_report(out))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    export(args.run, args.out)


if __name__ == "__main__":
    main()
