"""Synthetic journal fixtures only. No robot, policy, simulator, or agent execution."""

import argparse
import json
from pathlib import Path

from embodied_agent_harness.spec import expected_campaign, load_spec, write_json

BASE_REVISION = "a" * 40
CANDIDATE_REVISION = "b" * 40


def make_run(root: Path, spec: dict, revision: str, *, improved=False, regressed=False):
    root.mkdir(parents=True, exist_ok=False)
    write_json(
        root / "run.json",
        {"schema": "eah.run.v1", "evidence_kind": "fixture", "revision": revision, "spec": spec},
    )
    for panel, settings in spec["panels"].items():
        for seed in settings["seeds"]:
            directory = root / panel / f"seed-{seed}"
            manifest = expected_campaign(spec, panel, seed)
            manifest.update(
                campaign_id=f"fixture-{revision[:1]}-{panel}-{seed}",
                runtime={
                    "roborsi_commit": revision,
                    "roborsi_dirty": False,
                    "backend": spec["backend"],
                    "fixture": True,
                },
            )
            write_json(directory / "campaign.json", manifest)
            rows = []
            for task in settings["tasks"]:
                success = (
                    (improved or seed != settings["seeds"][0])
                    if (panel == "development")
                    else not regressed
                )
                rows.append(
                    {
                        "task_key": task,
                        "seed": seed,
                        "backend": spec["backend"],
                        "run_mode": "frozen",
                        "agent_mode": "roborsi",
                        "attempt": 1,
                        "run_id": f"fixture-{revision[:1]}-{task}-{seed}",
                        "status": "terminal",
                        "success": success,
                        "verdict": "success" if success else "failure",
                        "video_path": None,
                        "workspace": None,
                        "reviewer_verdict": "synthetic feedback; no real diagnosis",
                    }
                )
            (directory / "episodes.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=Path("examples/libero-small.json"))
    parser.add_argument("--out", type=Path, default=Path("runs/fixture-demo"))
    args = parser.parse_args()
    spec = load_spec(args.spec)
    make_run(args.out / "baseline", spec, BASE_REVISION)
    make_run(args.out / "candidate", spec, CANDIDATE_REVISION, improved=True)
    make_run(args.out / "regressed", spec, CANDIDATE_REVISION, improved=True, regressed=True)
    write_json(
        args.out / "proposal.json",
        {
            "schema": "eah.proposal.v1",
            "baseline_revision": BASE_REVISION,
            "candidate_revision": CANDIDATE_REVISION,
            "diagnosis_hypothesis": "Synthetic grasp failure; not inferred from real observations.",
            "change": "Synthetic candidate identity; no real skill has changed.",
            "rationale": "Exercise the paired regression reporting contract only.",
            "targets": [
                {
                    "task": spec["panels"]["development"]["tasks"][0],
                    "seed": spec["panels"]["development"]["seeds"][0],
                }
            ],
        },
    )
    print(f"FIXTURE ONLY: {args.out.resolve()}")


if __name__ == "__main__":
    main()
