"""Record a real LIBERO scene with a bounded scripted motion smoke test.

This is not an agent, trained policy, or task-solving benchmark. It exercises
reset, physics steps, observations, and video export on the local host.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.steps <= 1000:
        parser.error("steps must be between 1 and 1000")
    source, out = args.source.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    config_dir = out / "libero-config"
    config_dir.mkdir()
    benchmark_root = source / "libero/libero"
    import yaml

    paths = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "assets": str(benchmark_root / "assets"),
        "datasets": str(out),
    }
    (config_dir / "config.yaml").write_text(yaml.safe_dump(paths), encoding="utf-8")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    os.environ["MUJOCO_GL"] = "glfw"
    sys.path.insert(0, str(source))
    import imageio.v2 as imageio
    import numpy as np
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from PIL import Image, ImageDraw

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task)
    bddl = Path(suite.get_task_bddl_file_path(args.task))
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=384,
        camera_widths=512,
        camera_names=["agentview"],
        horizon=args.steps + 10,
        ignore_done=True,
    )
    trajectory, frames = [], []
    try:
        env.seed(args.seed)
        obs = env.reset()
        initial_success = bool(env.check_success())
        initial_eef = np.asarray(obs["robot0_eef_pos"]).copy()
        video = out / "preview.mp4"
        with imageio.get_writer(video, fps=20, codec="libx264", quality=8) as writer:
            for step in range(args.steps):
                # Small square in free space with open gripper; no object state access.
                phase = (step // 40) % 4
                action = np.zeros(7)
                axis, sign = [(0, 1), (1, 1), (0, -1), (1, -1)][phase]
                action[axis] = 0.18
                action[6] = -1.0
                obs, reward, done, info = env.step(action)
                success = bool(env.check_success())
                rgb = np.flipud(obs["agentview_image"]).copy()
                canvas = Image.new("RGB", (512, 432), (20, 25, 30))
                canvas.paste(Image.fromarray(rgb), (0, 48))
                draw = ImageDraw.Draw(canvas)
                draw.text(
                    (12, 6),
                    "REAL LIBERO / MuJoCo | SCRIPTED MOTION SMOKE TEST",
                    fill=(255, 255, 255),
                )
                draw.text(
                    (12, 25),
                    f"step {step + 1}/{args.steps} | seed {args.seed} | task success: {success}",
                    fill=(255, 255, 255),
                )
                frame = np.asarray(canvas)
                writer.append_data(frame)
                if step in {0, args.steps // 2, args.steps - 1}:
                    frames.append(Image.fromarray(frame))
                trajectory.append(
                    {
                        "step": step,
                        "action": action.tolist(),
                        "eef_pos": np.asarray(obs["robot0_eef_pos"]).tolist(),
                        "reward": float(reward),
                        "done": bool(done),
                        "simulator_success": success,
                    }
                )
                if step % 40 == 0:
                    print(f"rendered {step + 1}/{args.steps}", flush=True)
        final_eef = np.asarray(obs["robot0_eef_pos"])
        contact_sheet = Image.new("RGB", (512 * len(frames), 432))
        for index, frame in enumerate(frames):
            contact_sheet.paste(frame, (512 * index, 0))
        contact_sheet.save(out / "frames.png")
        (out / "trajectory.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in trajectory), encoding="utf-8"
        )
        result = {
            "evidence_kind": "simulator_smoke_test",
            "agent_used": False,
            "policy": "bounded_scripted_free_space_motion",
            "model_calls": 0,
            "suite": args.suite,
            "task_id": args.task,
            "instruction": task.language,
            "seed": args.seed,
            "initialization": "seeded_reset_not_benchmark_init_state",
            "steps": len(trajectory),
            "initial_success": initial_success,
            "final_success": bool(env.check_success()),
            "initial_eef_pos": initial_eef.tolist(),
            "final_eef_pos": final_eef.tolist(),
            "max_eef_displacement_m": float(
                max(np.linalg.norm(np.asarray(row["eef_pos"]) - initial_eef) for row in trajectory)
            ),
            "libero_revision": args.source_revision,
            "bddl_sha256": hashlib.sha256(bddl.read_bytes()).hexdigest(),
            "versions": {
                name: importlib.metadata.version(name)
                for name in ["mujoco", "robosuite", "numpy", "torch"]
            },
            "windows_compatibility": "see scripts/prepare_windows_preview.py",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "video": str(video),
            "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        }
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
