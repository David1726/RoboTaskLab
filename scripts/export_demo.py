"""Export selected actual demo evidence and media, removing machine-specific paths."""

import argparse
import shutil
from pathlib import Path

from embodied_agent_harness.spec import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reuse", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("assets/demo"))
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    run = args.run.resolve()
    for name in ["experiment.json", "history.json", "proposal-1.json", "accepted-skill.json"]:
        write_json(out / name, read_json(run / name))
    episodes = []
    originals = sorted(run.glob("*/result.json")) + [args.reuse.resolve() / "reuse/result.json"]
    for path in originals:
        record = read_json(path)
        if record.get("evidence_kind") != "simulator_execution":
            raise ValueError("Only real simulator execution evidence can be exported")
        name = path.parent.name
        destination = out / name
        destination.mkdir()
        for file in ["rollout.mp4", "trajectory.jsonl", "final.png"]:
            shutil.copyfile(path.parent / file, destination / file)
        record["video"] = f"{name}/rollout.mp4"
        write_json(destination / "result.json", record)
        episodes.append(
            {
                "episode": name,
                "success": record["success"],
                "split": record["split"],
                "init_state_index": record["seed"],
                "result": f"{name}/result.json",
                "video": record["video"],
            }
        )
    write_json(
        out / "summary.json",
        {
            "schema": "eah.demo_export.v1",
            "episodes": episodes,
            "source": "local actual LIBERO simulator execution; no model calls",
            "scope": "one task; controlled grasp-height defect; state-observing rule agent",
        },
    )
    print(out)


if __name__ == "__main__":
    main()
