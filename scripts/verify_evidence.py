"""Verify exported rollouts and provenance without rerunning or relabeling simulation."""

import argparse
import hashlib
import json
from pathlib import Path

from embodied_agent_harness.spec import read_json, strict_json, write_json
from embodied_agent_harness.transfer import paired_comparison


def verify(root):
    manifest = read_json(root / "export-manifest.json")
    for item in manifest["files"]:
        path = (root / item["file"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("export path escaped evidence root")
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["export_sha256"]:
            raise ValueError(f"export bytes changed: {item['file']}")
    targets, sources, versions = 0, 0, set()
    summaries = (
        list(root.glob("*/summary.json"))
        if (root / "report.json").exists()
        else [root / "summary.json"]
    )
    rows_by_condition = {}
    for path in summaries:
        rows = read_json(path)["rows"]
        is_source = path.parent.name == "source"
        if not is_source and (root / "report.json").exists():
            rows_by_condition[path.parent.name] = rows
        for row in rows:
            directory = path.parent / row["case"]["id"]
            if not directory.resolve().is_relative_to(root.resolve()):
                raise ValueError("episode outside evidence root")
            if read_json(directory / "result.json") != row:
                raise ValueError("episode result differs from summary")
            for field, name in [
                ("video_sha256", "rollout.mp4"),
                ("trajectory_sha256", "trajectory.jsonl"),
                ("initial_state_file_sha256", "initial-state.npy"),
            ]:
                if (
                    row.get(field)
                    and hashlib.sha256((directory / name).read_bytes()).hexdigest() != row[field]
                ):
                    raise ValueError(f"episode {field} differs from actual bytes")
            trajectory = [
                strict_json(line)
                for line in (directory / "trajectory.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            if len(trajectory) != row["steps"]:
                raise ValueError("trajectory length differs from executed steps")
            if (directory / "context.json").is_file():
                context = read_json(directory / "context.json")
                digest = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
                if digest != row["context_sha256"]:
                    raise ValueError("input context differs from recorded hash")
            versions.add(row.get("metric_version", "legacy_v1"))
            if is_source:
                sources += 1
            else:
                targets += 1
    if rows_by_condition:
        paired_comparison(rows_by_condition)
    return {
        "evidence": root.name,
        "target_episodes": targets,
        "source_episodes": sources,
        "export_files": len(manifest["files"]),
        "metric_versions": sorted(versions),
        "hashes_and_pairs_verified": True,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("roots", nargs="+", type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    results = [verify(root) for root in args.roots]
    if args.out:
        write_json(args.out, {"schema": "eah.evidence_verification.v1", "results": results})
    for result in results:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
