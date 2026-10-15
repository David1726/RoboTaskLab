"""Generate separate agent and cross-family Physical ICL manufacturing tracks."""

import argparse
import random
from pathlib import Path

from embodied_agent_harness.benchmark import load_panel
from embodied_agent_harness.language import manufacturing_instruction_task
from embodied_agent_harness.manufacturing import TASKS
from embodied_agent_harness.spec import write_json


def generate(
    seed, worlds, layouts, source_family, track, physical_axis="combined", instruction=None
):
    rng = random.Random(seed)
    if source_family == "random":
        source_family = rng.choice(["insertion", "nut_assembly", "stacking"])
    source_tasks = [t for t, (f, _) in TASKS.items() if f == source_family]
    source = rng.choice(source_tasks)
    source_layout = rng.randrange(2**31)
    source_opening = 0.014
    cases = []
    for world in range(worlds):
        physics = {
            "actuation_gain": rng.uniform(0.5, 1.5),
            "mass_scale": rng.uniform(0.65, 1.5),
            "friction": rng.uniform(0.3, 1.1),
        }
        if physical_axis == "actuation":
            physics.update(mass_scale=1.0, friction=0.8)
        elif physical_axis == "contact":
            physics["actuation_gain"] = 1.0
        target_tasks = [t for t, (f, _) in TASKS.items() if f != source_family]
        if track == "agent":
            target_tasks = list(TASKS)
            sources = list(TASKS)
        else:
            sources = [source]
        if instruction:
            target = manufacturing_instruction_task(instruction)
            if track == "physical_icl" and TASKS[target][0] == source_family:
                raise ValueError("language target family overlaps the Physical ICL source family")
            target_tasks = [target]
            if track == "agent":
                sources = [target]
        for split, tasks, count in [
            ("development", sources, 1),
            ("evaluation", target_tasks, layouts),
        ]:
            for task in tasks:
                for layout in range(count):
                    cases.append(
                        {
                            "id": f"{split}-w{world}-{task}-{layout}",
                            "backend": "manufacturing",
                            "task_id": task,
                            "split": split,
                            "task_family": TASKS[task][0],
                            "layout_mode": "seeded",
                            "layout_seed": source_layout
                            if split == "development"
                            else rng.randrange(2**31),
                            "world_id": f"world-{world}",
                            "physics": physics.copy(),
                            "opening_m": source_opening
                            if split == "development"
                            else rng.uniform(0.0135, 0.0155),
                        }
                    )
                    if instruction and (split == "evaluation" or track == "agent"):
                        cases[-1]["instruction"] = instruction
    rng.shuffle(cases)
    return {
        "schema": "eah.benchmark.v2",
        "name": f"Manufacturing {track} seeded suite",
        "evaluation_track": track,
        "observation_mode": "state",
        "max_steps": 1000,
        "max_skill_calls": 4,
        "generator_seed": seed,
        "source_family": source_family,
        "physical_worlds": worlds,
        "physical_axis": physical_axis,
        "source_counterfactual": "same source task and layout; physics differs across worlds",
        "cases": cases,
        "scope": "Single Panda contact simulation; actuator calibration + mass + friction. "
        "Not a neural ICL result or a real factory benchmark.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--worlds", type=int, default=2)
    parser.add_argument("--layouts", type=int, default=1)
    parser.add_argument(
        "--source-family",
        choices=["random", "insertion", "nut_assembly", "stacking"],
        default="nut_assembly",
    )
    parser.add_argument("--track", choices=["agent", "physical_icl"], default="physical_icl")
    parser.add_argument(
        "--physical-axis", choices=["combined", "actuation", "contact"], default="combined"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--instruction", help="supported explicit English/Chinese manufacturing goal"
    )
    args = parser.parse_args()
    if args.worlds < 2 or args.layouts < 1 or args.out.exists():
        parser.error("at least two worlds, positive layouts and a fresh output path required")
    write_json(
        args.out,
        generate(
            args.seed,
            args.worlds,
            args.layouts,
            args.source_family,
            args.track,
            args.physical_axis,
            args.instruction,
        ),
    )
    panel = load_panel(args.out)
    print(f"{args.out}: {len(panel['cases'])} cases; track={args.track}")


if __name__ == "__main__":
    main()
