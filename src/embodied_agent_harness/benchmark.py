"""Fixed-protocol benchmark runner with two participant-owned Python entry points."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import re
import shutil
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .language import instruction_task, manufacturing_instruction_task
from .libero_demo import create_env
from .manufacturing import TASKS
from .metrics import (
    METRIC_VERSION,
    SUCCESS_HOLD_STEPS,
    force_assessment,
    task_metric_contract,
    validate_force_limit,
)
from .spec import read_json, write_json


class BudgetExceeded(RuntimeError):
    pass


def runtime_versions(packages):
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path.resolve())
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(path.read_bytes(), str(path.resolve()), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def verify_frozen(identities):
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p, h in identities.items()):
        raise ValueError("frozen participant code, protocol or panel changed during evaluation")


def protocol_files():
    return sorted(Path(__file__).parent.glob("*.py"))


@contextmanager
def campaign_state(root):
    manifest = read_json(root / "manifest.json")
    manifest["status"] = "running"
    write_json(root / "manifest.json", manifest)
    try:
        yield
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(root / "manifest.json", manifest)
        raise
    else:
        manifest["status"] = "completed"
        write_json(root / "manifest.json", manifest)


def execution_identity():
    return {
        "core_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in protocol_files()
        },
        "versions": runtime_versions(["mujoco", "robosuite", "numpy"]),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform": sys.platform,
    }


def load_panel(path: Path) -> dict:
    panel = read_json(path)
    if (
        panel.get("schema") not in {"eah.benchmark.v1", "eah.benchmark.v2"}
        or panel.get("observation_mode") != "state"
    ):
        raise ValueError("benchmark v1/v2 and declared state observations required")
    for field in ["max_steps", "max_skill_calls"]:
        if type(panel.get(field)) is not int or panel[field] < 1:
            raise ValueError(f"invalid {field}")
    if panel.get("schema") == "eah.benchmark.v2" and panel.get("evaluation_track") not in {
        "agent",
        "physical_icl",
    }:
        raise ValueError("v2 requires an explicit agent or physical_icl track")
    cases = panel.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("nonempty cases required")
    ids, pairs = set(), set()
    worlds = {}
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("cases must be JSON objects")
        if case.get("backend", "libero") not in {"libero", "manufacturing"}:
            raise ValueError("unsupported backend")
        if (
            panel.get("evaluation_track") == "physical_icl"
            and case.get("backend") != "manufacturing"
        ):
            raise ValueError("physical_icl v2 requires defined manufacturing physical worlds")
        if (
            not isinstance(case.get("id"), str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", case["id"]) is None
            or case["id"] in ids
        ):
            raise ValueError("case IDs must be unique safe directory names")
        if case["id"].upper() in {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            *[f"COM{i}" for i in range(1, 10)],
            *[f"LPT{i}" for i in range(1, 10)],
        }:
            raise ValueError("case ID is a reserved device name")
        if case.get("backend") == "manufacturing":
            if (
                not isinstance(case.get("task_id"), str)
                or case.get("task_id") not in TASKS
                or case.get("split") not in {"development", "evaluation"}
                or type(case.get("layout_seed")) is not int
                or not 0 <= case["layout_seed"] < 2**32
                or case.get("layout_mode") != "seeded"
                or not isinstance(case.get("world_id"), str)
                or not case["world_id"].strip()
            ):
                raise ValueError("invalid manufacturing task/layout/world")
            physics = case.get("physics", {})
            if not isinstance(physics, dict):
                raise ValueError("physics must be a mapping")
            for field, bounds in {
                "actuation_gain": (0.35, 2),
                "mass_scale": (0.5, 2),
                "friction": (0.2, 1.5),
            }.items():
                value = physics.get(field)
                if type(value) not in (int, float) or not bounds[0] <= value <= bounds[1]:
                    raise ValueError("invalid manufacturing physics")
            if (
                type(case.get("opening_m", 0.014)) not in (int, float)
                or not 0.012 <= case.get("opening_m", 0.014) <= 0.020
            ):
                raise ValueError("invalid insertion opening")
            validate_force_limit(case)
            previous = worlds.setdefault(case["world_id"], physics)
            if previous != physics:
                raise ValueError("one physical world must have consistent physics")
            if (
                "instruction" in case
                and manufacturing_instruction_task(case["instruction"]) != case["task_id"]
            ):
                raise ValueError("manufacturing language does not match goal")
            case["task_family"] = TASKS[case["task_id"]][0]
            pair = ("manufacturing", case["task_id"], case["world_id"], case["layout_seed"])
            if pair in pairs:
                raise ValueError("task/layout pairs overlap")
            pairs.add(pair)
            ids.add(case["id"])
            continue
        if (
            case.get("split") not in {"development", "evaluation"}
            or type(case.get("task_id")) is not int
            or not 0 <= case["task_id"] < 10
        ):
            raise ValueError("invalid task/init-state/split")
        if case.get("layout_mode", "official") == "official":
            if (
                type(case.get("init_state_index")) is not int
                or not 0 <= case["init_state_index"] < 50
            ):
                raise ValueError("invalid official init-state index")
            pair = (case["task_id"], "official", case["init_state_index"])
        elif case["layout_mode"] == "expanded_reset":
            if (
                type(case.get("layout_seed")) is not int
                or not 0 <= case["layout_seed"] < 2**32
                or type(case.get("jitter_m")) not in (float, int)
                or not 0 < case["jitter_m"] <= 0.08
            ):
                raise ValueError("invalid randomized layout settings")
            pair = (case["task_id"], "randomized", case["layout_seed"])
        else:
            raise ValueError("unknown layout mode")
        if "instruction" in case and instruction_task(case["instruction"]) != case["task_id"]:
            raise ValueError("language instruction differs from simulator task goal")
        if pair in pairs:
            raise ValueError("task/init-state pairs must not repeat or overlap splits")
        ids.add(case["id"])
        pairs.add(pair)
    return panel


def summarize(rows):
    if not rows:
        raise ValueError("cannot summarize an empty benchmark")
    return {
        "schema": "eah.benchmark_summary.v1",
        "episodes": len(rows),
        "successes": sum(r["success"] for r in rows),
        "success_rate": sum(r["success"] for r in rows) / len(rows),
        "implementation_errors": sum(r["termination"] == "implementation_error" for r in rows),
        "by_split": {
            split: {
                "episodes": len(group),
                "successes": sum(r["success"] for r in group),
                "success_rate": sum(r["success"] for r in group) / len(group),
            }
            for split in sorted({r["case"]["split"] for r in rows})
            if (group := [r for r in rows if r["case"]["split"] == split])
        },
        "by_task": {
            str(task): {
                "episodes": len(group),
                "successes": sum(r["success"] for r in group),
                "success_rate": sum(r["success"] for r in group) / len(group),
            }
            for task in sorted(
                {r["case"]["task_id"] for r in rows if "task_id" in r["case"]}, key=str
            )
            if (group := [r for r in rows if r["case"].get("task_id") == task])
        },
        "rows": rows,
    }


def validate_action(action):
    if not isinstance(action, (list, tuple)) or len(action) != 7:
        raise ValueError("step requires seven normalized action values")
    if any(type(v) not in (int, float) or not math.isfinite(v) or not -1 <= v <= 1 for v in action):
        raise ValueError("actions must contain finite values in [-1,1]")


class SkillContext:
    """Participant control surface. No reward, done, or goal oracle in observations."""

    def __init__(
        self, env, observation, instruction, root, max_steps, writer, experience_context=None
    ):
        self._env, self._obs = env, observation
        self._instruction, self._root = instruction, root
        self._limit, self._writer = max_steps, writer
        self.steps = 0
        self._experience_context = experience_context or {"mode": "no_context", "experiences": []}
        self._trajectory = []
        self.peak_wrist_force = None
        self._success_tail = 0
        self._frame_step = None
        # Actor paths must not expose target world/case/experimental condition names.
        self._actor_frame_base = Path.cwd().resolve() / ".cache" / "actor-frames"
        self._actor_frame_base.mkdir(parents=True, exist_ok=True)
        self._actor_frame = self._actor_frame_base / f"{uuid.uuid4().hex}.png"

    def observe(self):
        import numpy as np
        from PIL import Image

        frame = self._root / "latest.png"
        if self._frame_step != self.steps:
            Image.fromarray(np.flipud(self._obs["agentview_image"]).copy()).save(frame)
            shutil.copyfile(frame, self._actor_frame)
            self._frame_step = self.steps
        return {
            "schema": "eah.observation.v1",
            "instruction": self._instruction,
            "eef_pos": self._obs["robot0_eef_pos"].tolist(),
            "eef_quat_xyzw": self._obs["robot0_eef_quat"].tolist(),
            "gripper_qpos": self._obs["robot0_gripper_qpos"].tolist(),
            "objects": {
                name: self._env.sim.data.body_xpos[self._env.env.obj_body_id[name]].tolist()
                for name in self._env.env.objects_dict
            },
            "image_path": str(self._actor_frame),
            "remaining_steps": self._limit - self.steps,
            "context": json.loads(json.dumps(self._experience_context)),
            **(self._env.public_observation() if hasattr(self._env, "public_observation") else {}),
        }

    def step(self, action):
        import numpy as np
        from PIL import Image, ImageDraw

        validate_action(action)
        if self.steps >= self._limit:
            raise BudgetExceeded("episode action budget exhausted")
        self._obs, reward, done, info = self._env.step(action)
        self.steps += 1
        record = {"step": self.steps, "action": list(action), "observation": None}
        self._trajectory.append(record)
        if getattr(self._env, "case", {}).get("backend") == "manufacturing":
            self._success_tail = self._success_tail + 1 if self._env.check_success() else 0
        observation = self.observe()
        trace_observation = {
            k: v for k, v in observation.items() if k not in {"image_path", "context"}
        }
        json.dumps(trace_observation, allow_nan=False)
        record["observation"] = trace_observation
        if "wrist_force_N" in observation:
            force = math.sqrt(sum(v * v for v in observation["wrist_force_N"]))
            self.peak_wrist_force = max(self.peak_wrist_force or 0, force)
        # Images are recorded every two 20 Hz control ticks, hence 10 fps playback.
        if self.steps % 2 == 0:
            image = Image.new("RGB", (512, 432), (20, 25, 30))
            image.paste(Image.fromarray(np.flipud(self._obs["agentview_image"]).copy()), (0, 48))
            draw = ImageDraw.Draw(image)
            draw.text((10, 8), "EMBODIED BENCHMARK | STATE OBSERVATIONS", fill="white")
            draw.text((10, 28), f"step {self.steps}/{self._limit}", fill="white")
            self._writer.append_data(np.asarray(image))
        return observation

    def save(self):
        (self._root / "trajectory.jsonl").write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in self._trajectory),
            encoding="utf-8",
        )
        if self._actor_frame.resolve().parent != self._actor_frame_base:
            raise ValueError("actor frame cleanup escaped its designated cache directory")
        self._actor_frame.unlink(missing_ok=True)


def execute_case(
    env, task, states, case, panel, skills, agent_module, root, experience_context=None
):
    import imageio.v2 as imageio

    if hasattr(env, "initialize"):
        obs = env.initialize(case)
    elif case.get("layout_mode", "official") == "expanded_reset":
        # Preserve the configured samplers across reset instead of reconstructing the model.
        env.env.hard_reset = False
        for sampler in env.env.placement_initializer.samplers.values():
            if not all(obj.name in env.env.objects_dict for obj in sampler.mujoco_objects):
                continue
            for field in ["x_ranges", "y_ranges"]:
                ranges = getattr(sampler, field)
                # Sample centers in wider bounded regions; retain upstream collision rejection.
                setattr(
                    sampler,
                    field,
                    [
                        [(lo + hi) / 2 - case["jitter_m"], (lo + hi) / 2 + case["jitter_m"]]
                        for lo, hi in ranges
                    ],
                )
            sampler.ensure_object_boundary_in_range = False
        env.seed(case["layout_seed"])
        obs = env.reset()
    else:
        env.seed(0)
        env.reset()
        obs = env.set_init_state(states[case["init_state_index"]])
    import numpy as np

    np.save(root / "initial-state.npy", env.get_sim_state(), allow_pickle=False)
    initial_hash = hashlib.sha256(env.get_sim_state().tobytes()).hexdigest()
    if env.check_success():
        raise ValueError("task is already solved at initialization")
    feedback = None
    calls, error, termination = [], None, "skill_call_budget"
    started = time.monotonic()
    with imageio.get_writer(root / "rollout.mp4", fps=10, codec="libx264", quality=7) as writer:
        import numpy as np
        from PIL import Image, ImageDraw

        initial_image = Image.new("RGB", (512, 432), (20, 25, 30))
        initial_image.paste(Image.fromarray(np.flipud(obs["agentview_image"]).copy()), (0, 48))
        ImageDraw.Draw(initial_image).text((10, 12), "SEEDED ROBOT TASK EXECUTION", fill="white")
        writer.append_data(np.asarray(initial_image))
        Image.fromarray(np.flipud(obs["agentview_image"]).copy()).save(root / "latest.png")
        context = SkillContext(
            env,
            obs,
            case.get("instruction", task.language),
            root,
            panel["max_steps"],
            writer,
            experience_context,
        )
        write_json(root / "context.json", context._experience_context)
        try:
            if isinstance(skills, Path):
                skills = load_module(skills, "episode_skills")
            if isinstance(agent_module, Path):
                agent_module = load_module(agent_module, "episode_agent")
            agent = agent_module.create_agent()
            for index in range(panel["max_skill_calls"]):
                decision = agent.decide(context.observe(), feedback)
                if not isinstance(decision, dict):
                    raise ValueError("agent.decide must return a decision dict")
                if decision.get("type") == "stop":
                    termination = "agent_stop"
                    break
                if (
                    decision.get("type") != "skill"
                    or not isinstance(decision.get("name"), str)
                    or not decision["name"].strip()
                    or not isinstance(decision.get("arguments"), dict)
                ):
                    raise ValueError("decision must be stop or a named skill with arguments")
                # Save an immutable serializable copy before participant code can mutate it.
                record = {
                    "index": index,
                    "decision": json.loads(json.dumps(decision, allow_nan=False)),
                    "start_step": context.steps,
                }
                calls.append(record)
                try:
                    feedback = skills.execute(decision, context)
                    if not isinstance(feedback, dict) or not isinstance(
                        feedback.get("status"), str
                    ):
                        raise ValueError("skills.execute must return feedback with a status string")
                    record["feedback"] = json.loads(json.dumps(feedback, allow_nan=False))
                finally:
                    record["end_step"] = context.steps
        except BudgetExceeded as exc:
            termination, error = "step_budget", str(exc)
        except Exception as exc:
            termination, error = "implementation_error", f"{type(exc).__name__}: {exc}"
        finally:
            context.save()
        # The evaluator owns this predicate; never hand it to participant decide/execute.
        simulator_success = bool(env.check_success())
        result = {
            "schema": "eah.benchmark_episode.v1",
            "evidence_kind": "simulator_execution",
            "case": case,
            "success": simulator_success and termination != "implementation_error",
            "simulator_success": simulator_success,
            "termination": termination,
            "error": error,
            "steps": context.steps,
            "skill_calls": len(calls),
            "wall_seconds": time.monotonic() - started,
            "initial_state_sha256": initial_hash,
            "instruction": case.get("instruction", task.language),
            "context_mode": context._experience_context["mode"],
            "context_bytes": len(
                json.dumps(context._experience_context, ensure_ascii=False).encode()
            ),
            "context_sha256": hashlib.sha256(
                json.dumps(context._experience_context, sort_keys=True).encode()
            ).hexdigest(),
            "calls": calls,
            "peak_wrist_force_N": context.peak_wrist_force,
            "evaluation_metrics": env.evaluation_metrics()
            if hasattr(env, "evaluation_metrics")
            else {},
            "video": str(root / "rollout.mp4"),
        }
    result["video_sha256"] = hashlib.sha256((root / "rollout.mp4").read_bytes()).hexdigest()
    result["trajectory_sha256"] = hashlib.sha256(
        (root / "trajectory.jsonl").read_bytes()
    ).hexdigest()
    result["initial_state_file_sha256"] = hashlib.sha256(
        (root / "initial-state.npy").read_bytes()
    ).hexdigest()
    if case.get("backend") == "manufacturing":
        result["task_metric_contract"] = task_metric_contract(case)
        result["final_success_hold_steps"] = context._success_tail
        result["success"] = result["success"] and context._success_tail >= SUCCESS_HOLD_STEPS
        result["simulated_seconds"] = context.steps / 20
        result.update(force_assessment(case, result["success"], context.peak_wrist_force))
    write_json(root / "result.json", result)
    return result


def create_case_env(source, directory, case):
    if case.get("backend") == "manufacturing":
        from .manufacturing import create_env as create_manufacturing

        return create_manufacturing(case, directory)
    if source is None:
        raise ValueError("LIBERO cases require --source")
    return create_env(source.resolve(), directory, case["task_id"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--panel", type=Path, default=Path("benchmarks/libero-object-pilot.json"))
    parser.add_argument("--skills", type=Path, default=Path("participant/skills.py"))
    parser.add_argument("--agent", type=Path, default=Path("participant/agent.py"))
    parser.add_argument(
        "--split", choices=["development", "evaluation", "all"], default="development"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--instruction", help="language goal; must match each selected case")
    args = parser.parse_args(argv)
    panel = load_panel(args.panel)
    selected = [c for c in panel["cases"] if args.split == "all" or c["split"] == args.split]
    if not selected:
        parser.error("selected split contains no cases")
    if args.source is None and any(c.get("backend", "libero") == "libero" for c in selected):
        parser.error("LIBERO cases require --source")
    if args.instruction:
        selected = [c for c in panel["cases"] if args.split == "all" or c["split"] == args.split]
        manufacturing = all(c.get("backend") == "manufacturing" for c in selected)
        task_id = (
            manufacturing_instruction_task(args.instruction)
            if manufacturing
            else instruction_task(args.instruction)
        )
        if any(c["task_id"] != task_id for c in selected):
            parser.error("instruction does not match panel tasks; generate a panel for that goal")
        for case in selected:
            case["instruction"] = args.instruction
    root = args.out.resolve()
    identities = {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [args.skills, args.agent, args.panel]
    }
    frozen = {
        **identities,
        **{str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protocol_files()},
    }
    root.mkdir(parents=True, exist_ok=False)
    write_json(
        root / "manifest.json",
        {
            "panel": panel,
            "metric_version": METRIC_VERSION,
            "execution_identity": execution_identity(),
            "split": args.split,
            "participant_sha256": identities,
            "entrypoint_sha256": {
                "skills": identities[str(args.skills.resolve())],
                "agent": identities[str(args.agent.resolve())],
            },
            "runtime": {
                "python": sys.version,
                "platform": sys.platform,
                "source_path": str(args.source.resolve()) if args.source else None,
                "versions": runtime_versions(["mujoco", "robosuite", "numpy", "torch", "imageio"]),
                "control": "Panda OSC_POSE, normalized 7D, 20Hz",
            },
            "trust": "trusted Python plugins; not an adversarial sandbox",
        },
    )
    with campaign_state(root):
        rows = []
        for case in panel["cases"]:
            if args.split != "all" and case["split"] != args.split:
                continue
            verify_frozen(frozen)
            directory = root / case["id"]
            directory.mkdir()
            # Reset modules and the instance so globals do not carry target experience.
            env, task, states, init_hash = create_case_env(args.source, directory, case)
            try:
                row = execute_case(
                    env, task, states, case, panel, args.skills, args.agent, directory
                )
                verify_frozen(frozen)
                row["init_asset_sha256"] = init_hash
                write_json(directory / "result.json", row)
                rows.append(row)
                summary = summarize(rows)
                write_json(root / "summary.json", summary)
                print(f"{case['id']}: success={row['success']}, steps={row['steps']}", flush=True)
            finally:
                env.close()
        if rows:
            from .report import render_benchmark_report

            render_benchmark_report(root)
        return 2 if not rows else int(any(not row["success"] for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
