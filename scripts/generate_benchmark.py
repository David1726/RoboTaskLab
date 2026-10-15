"""Sample a reproducible randomized LIBERO-derived pilot panel and language goals."""

import argparse
import random
from pathlib import Path

from embodied_agent_harness.language import instruction_task
from embodied_agent_harness.spec import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--development", type=int, default=1)
    parser.add_argument("--evaluation", type=int, default=3)
    parser.add_argument("--tasks", type=int, nargs="+", default=[0, 1, 5])
    parser.add_argument("--source-tasks", type=int, nargs="+")
    parser.add_argument("--target-tasks", type=int, nargs="+")
    parser.add_argument("--instruction", help="optional supported English/Chinese pick-place goal")
    parser.add_argument("--jitter-m", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.development < 1 or args.evaluation < 1 or not 0 < args.jitter_m <= 0.08:
        parser.error("positive split sizes and jitter in (0,0.08] required")
    tasks = [instruction_task(args.instruction)] if args.instruction else args.tasks
    if bool(args.source_tasks) != bool(args.target_tasks):
        parser.error("source and target task pools must be supplied together")
    if args.source_tasks and (args.instruction or set(args.source_tasks) & set(args.target_tasks)):
        parser.error("cross-task pools must be disjoint and cannot override one language goal")
    if args.source_tasks:
        tasks = args.source_tasks + args.target_tasks
    if any(not 0 <= task < 10 for task in tasks):
        parser.error("libero_object task IDs must be 0..9")
    if args.out.exists():
        parser.error("panel already exists; choose a new path")
    rng, cases = random.Random(args.seed), []
    for split, count in [("development", args.development), ("evaluation", args.evaluation)]:
        pool = (
            (args.source_tasks if split == "development" else args.target_tasks)
            if args.source_tasks
            else tasks
        )
        for index in range(count):
            case = {
                "id": f"{split}-{index}",
                "split": split,
                "task_id": rng.choice(pool),
                "layout_seed": rng.randrange(2**31),
                "layout_mode": "expanded_reset",
                "jitter_m": args.jitter_m,
            }
            if args.instruction:
                case["instruction"] = args.instruction
            cases.append(case)
    write_json(
        args.out,
        {
            "schema": "eah.benchmark.v1",
            "name": "Randomized LIBERO-derived pilot",
            "generator_seed": args.seed,
            "observation_mode": "state",
            "max_steps": 600,
            "max_skill_calls": 3,
            "cases": cases,
            "scope": "custom placement distribution; fixed Panda; not official LIBERO scores",
        },
    )
    print(args.out.resolve())


if __name__ == "__main__":
    main()
