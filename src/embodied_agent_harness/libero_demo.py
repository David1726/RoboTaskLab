"""Real-simulator, state-observing rule-agent RSI demonstration on one LIBERO task."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import uuid
from dataclasses import asdict
from pathlib import Path

from .agents import checked_proposal, load_agent
from .rsi import PickPlaceSkill, gate
from .spec import read_json, write_json


def create_env(source: Path, root: Path, task_id: int = 0):
    import yaml

    config = root / "libero-config"
    config.mkdir()
    assets = source / "libero/libero"
    paths = {
        "benchmark_root": str(assets),
        "bddl_files": str(assets / "bddl_files"),
        "init_states": str(assets / "init_files"),
        "assets": str(assets / "assets"),
        "datasets": str(root),
    }
    (config / "config.yaml").write_text(yaml.safe_dump(paths), encoding="utf-8")
    os.environ["LIBERO_CONFIG_PATH"] = str(config)
    if sys.platform == "win32":
        os.environ["MUJOCO_GL"] = "glfw"
    sys.path.insert(0, str(source))
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    task = suite.get_task(task_id)
    env = OffScreenRenderEnv(
        bddl_file_name=suite.get_task_bddl_file_path(task_id),
        camera_heights=384,
        camera_widths=512,
        camera_names=["agentview"],
        horizon=650,
        ignore_done=True,
    )
    import torch

    init_path = assets / "init_files" / task.problem_folder / task.init_states_file
    # Load only the trusted official source asset; torch's legacy format uses pickle.
    states = torch.load(init_path, weights_only=False, map_location="cpu")
    return env, task, states, hashlib.sha256(init_path.read_bytes()).hexdigest()


def rollout(env, skill: PickPlaceSkill, seed: int, split: str, directory: Path, states) -> dict:
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw

    directory.mkdir()
    env.seed(seed)
    env.reset()
    obs = env.set_init_state(states[seed])
    initial_hash = hashlib.sha256(env.get_sim_state().tobytes()).hexdigest()
    initial_success = bool(env.check_success())
    if initial_success:
        raise ValueError("task already succeeds before execution")
    target_name, basket_name = "alphabet_soup_1", "basket_1"

    def position(name):
        return env.sim.data.body_xpos[env.env.obj_body_id[name]].copy()

    target_initial = position(target_name)
    basket_initial = position(basket_name)
    events, trajectory = [], []
    failure = None
    close_clearance, close_xy = None, None
    video_path = directory / "rollout.mp4"
    with imageio.get_writer(video_path, fps=10, codec="libx264", quality=7) as writer:

        def move(phase, goal, gripper, max_steps=75, min_steps=5):
            nonlocal obs
            for local in range(max_steps):
                error = goal - np.asarray(obs["robot0_eef_pos"])
                action = np.zeros(7)
                action[:3] = np.clip(error * 12, -0.6, 0.6)
                action[6] = gripper
                obs, reward, done, info = env.step(action)
                target = position(target_name)
                eef = np.asarray(obs["robot0_eef_pos"])
                trajectory.append(
                    {
                        "step": len(trajectory),
                        "phase": phase,
                        "action": action.tolist(),
                        "eef": eef.tolist(),
                        "object": target.tolist(),
                        "reward": float(reward),
                    }
                )
                if len(trajectory) % 2 == 0:
                    frame = Image.new("RGB", (512, 432), (20, 25, 30))
                    frame.paste(Image.fromarray(np.flipud(obs["agentview_image"]).copy()), (0, 48))
                    draw = ImageDraw.Draw(frame)
                    draw.text(
                        (10, 5), "LIBERO | STATE-OBSERVING RULE AGENT | PARAMETER RSI", fill="white"
                    )
                    draw.text(
                        (10, 25),
                        f"{split} seed={seed} | {phase} | grasp offset={skill.grasp_offset_m:.3f}m",
                        fill="white",
                    )
                    writer.append_data(np.asarray(frame))
                if local >= min_steps and np.linalg.norm(goal - eef) < 0.008:
                    break
            events.append(
                {
                    "phase": phase,
                    "steps": local + 1,
                    "goal_error_m": float(np.linalg.norm(goal - eef)),
                }
            )
            print(f"  {phase}: error={np.linalg.norm(goal - eef):.4f}", flush=True)

        approach = np.array([*target_initial[:2], skill.transport_height_m])
        grasp = target_initial + [0, 0, skill.grasp_offset_m]
        move("approach", approach, -1)
        move("descend", grasp, -1)
        move("close", grasp, 1, max_steps=30, min_steps=29)
        close_eef = np.asarray(obs["robot0_eef_pos"])
        close_obj = position(target_name)
        close_clearance = float(close_eef[2] - close_obj[2])
        close_xy = float(np.linalg.norm(close_eef[:2] - close_obj[:2]))
        move("lift", approach, 1)
        object_lift = float(position(target_name)[2] - target_initial[2])
        if object_lift < 0.045:
            failure = "empty_grasp"
        else:
            destination = np.array([*basket_initial[:2], skill.transport_height_m])
            move("transport", destination, 1, max_steps=100)
            release = np.array([*basket_initial[:2], 0.16])
            move("lower", release, 1)
            move("release", release, -1, max_steps=35, min_steps=34)
            move("retreat", destination, -1)
            move("settle", destination, -1, max_steps=35, min_steps=34)
    # Only the evaluator reads the task predicate after the executor has stopped.
    success = bool(env.check_success())
    if failure is None and not success:
        failure = "placement_failed"
    Image.fromarray(np.flipud(obs["agentview_image"]).copy()).save(directory / "final.png")
    journal = directory / "trajectory.jsonl"
    journal.write_text("".join(json.dumps(r) + "\n" for r in trajectory), encoding="utf-8")
    result = {
        "run_id": uuid.uuid4().hex,
        "evidence_kind": "simulator_execution",
        "split": split,
        "seed": seed,
        "skill_id": skill.identity,
        "skill": asdict(skill),
        "success": success,
        "initial_success": initial_success,
        "failure": failure,
        "initial_state_sha256": initial_hash,
        "initial_object_pos": target_initial.tolist(),
        "initial_basket_pos": basket_initial.tolist(),
        "steps": len(trajectory),
        "events": events,
        "feedback": {
            "failure": failure,
            "object_lift_m": object_lift,
            "close_vertical_clearance_m": close_clearance,
            "close_xy_error_m": close_xy,
        },
        "video": str(video_path),
        "video_sha256": hashlib.sha256(video_path.read_bytes()).hexdigest(),
        "trajectory_sha256": hashlib.sha256(journal.read_bytes()).hexdigest(),
        "observation_scope": "privileged simulator object positions and robot proprioception",
    }
    write_json(directory / "result.json", result)
    print(f"{split} seed={seed}: success={success}, failure={failure}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skill", type=Path, help="reuse an accepted skill; skip refinement")
    parser.add_argument(
        "--seed", type=int, default=0, help="official init-state index for skill reuse"
    )
    parser.add_argument("--max-revisions", type=int, default=2)
    parser.add_argument("--agent", default="height-repair", help="height-repair or module:factory")
    args = parser.parse_args(argv)
    if not 1 <= args.max_revisions <= 3:
        parser.error("max-revisions must be 1..3")
    agent = load_agent(args.agent)
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    env, task, states, init_hash = create_env(args.source.resolve(), root)
    try:
        manifest = {
            "schema": "eah.rsi_demo.v1",
            "task": "libero_object/0",
            "instruction": task.language,
            "agent": args.agent,
            "revision_type": "skill_parameter_refinement",
            "initialization": "official_LIBERO_init_state_indices",
            "init_asset_sha256": init_hash,
            "development_seeds": [0],
            "heldout_seeds": [1, 2],
            "scope": "same task, heldout layouts only; not heldout tasks",
            "max_revisions": args.max_revisions,
            "model_calls": 0,
            "versions": {
                p: importlib.metadata.version(p) for p in ["mujoco", "robosuite", "numpy"]
            },
        }
        write_json(root / "experiment.json", manifest)
        if args.skill:
            if not 0 <= args.seed < len(states):
                raise ValueError("init-state index outside official asset")
            saved = read_json(args.skill)
            if (
                saved.get("schema") != "eah.accepted_skill.v1"
                or saved.get("task") != "libero_object/0"
            ):
                raise ValueError("skill is not an accepted LIBERO demo skill")
            skill = PickPlaceSkill(**saved["parameters"])
            result = rollout(env, skill, args.seed, "reuse", root / "reuse", states)
            return 0 if result["success"] else 1
        skill = PickPlaceSkill()
        baseline = [rollout(env, skill, 0, "development", root / "baseline-development-0", states)]
        history = []
        working_skill, feedback = skill, baseline[0]["feedback"]
        for revision in range(1, args.max_revisions + 1):
            hypothesis = checked_proposal(agent, working_skill, feedback)
            if hypothesis is None:
                break
            candidate, proposal = hypothesis
            write_json(root / f"proposal-{revision}.json", proposal)
            trial = rollout(
                env,
                candidate,
                0,
                "development",
                root / f"candidate-{revision}-development-0",
                states,
            )
            history.append(
                {
                    "revision": revision,
                    "proposal": proposal,
                    "development_success": trial["success"],
                }
            )
            write_json(root / "history.json", {"revisions": history})
            if not trial["success"]:
                working_skill, feedback = candidate, trial["feedback"]
                continue
            # Freeze candidate before any held-out evidence is evaluated. No tuning after this gate.
            baseline += [
                rollout(env, skill, seed, "heldout", root / f"baseline-heldout-{seed}", states)
                for seed in [1, 2]
            ]
            results = [trial] + [
                rollout(
                    env,
                    candidate,
                    seed,
                    "heldout",
                    root / f"candidate-{revision}-heldout-{seed}",
                    states,
                )
                for seed in [1, 2]
            ]
            decision = gate(baseline, results)
            history[-1]["gate"] = decision
            write_json(root / "history.json", {"revisions": history})
            if decision["accepted"]:
                write_json(
                    root / "accepted-skill.json",
                    {
                        "schema": "eah.accepted_skill.v1",
                        "task": "libero_object/0",
                        "skill_id": candidate.identity,
                        "parameters": asdict(candidate),
                        "gate": decision,
                    },
                )
                print(json.dumps(decision, indent=2), flush=True)
                return 0
            # Stop after held-out evaluation; rejected candidates are not promoted or retuned.
            break
        print("No candidate accepted; inspect real trajectory evidence.", flush=True)
        return 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
