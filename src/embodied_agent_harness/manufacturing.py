"""Single-Panda manufacturing tasks and hidden physical-world interventions.

Peg insertion uses local contact geometry and a local geometric/release oracle.
Nut assembly and stacking use robosuite 1.4 tasks and their upstream oracles.
No object pose is changed after initialization by any participant action.
"""

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

TASKS = {
    "insert_round": ("insertion", "Insert the round peg into the round hole and release it."),
    "insert_square": ("insertion", "Insert the square peg into the square hole and release it."),
    "nut_round": ("nut_assembly", "Place the round nut onto its shaft and release it."),
    "nut_square": ("nut_assembly", "Place the square nut onto its shaft and release it."),
    "stack": ("stacking", "Stack the small block on the large block and release it."),
}


def insertion_success(
    position, axis, hole, half_length, radius, clearance, released, shape="round", rotation=None
):
    """Task-defined depth/release/uprightness plus cross-section containment."""
    tip = [p - half_length * a for p, a in zip(position, axis, strict=True)]
    lateral = math.dist(tip[:2], hole[:2])
    depth = hole[2] - tip[2]
    geometric = bool(
        released and axis[2] > 0.995 and lateral < clearance - radius and 0.018 <= depth <= 0.034
    )
    if not geometric:
        return False
    if shape == "square" and rotation is None:
        raise ValueError("square insertion requires orientation for footprint assessment")
    return (
        insertion_fit_margin(
            position, axis, hole, radius, clearance, shape, rotation, half_length=half_length
        )
        > 0
    )


def insertion_fit_margin(
    position, axis, hole, radius, clearance, shape, rotation=None, half_length=0.05
):
    """Conservative prism fit at entry and lowest fully engaged wall cross-section.

    Round fit uses a bounding circle for the tilted elliptical section. Square
    fit projects all four prism corners, including yaw. This is a geometric
    oracle for this coarse fixture, not a claim of precision assembly quality.
    """
    if axis[2] <= 0:
        return None
    tip_z = position[2] - half_length * axis[2]
    tilt_extent = radius * math.sqrt(2) * math.sqrt(max(0, 1 - axis[2] ** 2))
    lower = max(hole[2] - 0.032, tip_z + tilt_extent + 1e-6)
    margins = []
    for plane in [hole[2], min(hole[2], lower)]:
        t = (plane - position[2]) / axis[2]
        center = [position[i] + t * axis[i] - hole[i] for i in range(2)]
        if shape == "round":
            margins.append(clearance - math.hypot(*center) - radius / axis[2])
        elif shape == "square":
            if rotation is None:
                raise ValueError("square footprint requires orientation")
            for u in [-radius, radius]:
                for v in [-radius, radius]:
                    axial = (plane - position[2] - rotation[2][0] * u - rotation[2][1] * v) / axis[
                        2
                    ]
                    for i in range(2):
                        corner = (
                            position[i] + rotation[i][0] * u + rotation[i][1] * v + axial * axis[i]
                        )
                        margins.append(clearance - abs(corner - hole[i]))
        else:
            raise ValueError("unsupported insertion shape")
    return min(margins)


def fixture_xml(shape, opening):
    geoms = []
    if shape == "round":
        for i in range(24):
            angle = 2 * math.pi * i / 24
            radial = opening + 0.01
            x, y = radial * math.cos(angle), radial * math.sin(angle)
            tangent = radial * math.tan(math.pi / 24) + 0.001
            geoms.append(
                f'<geom type="box" pos="{x} {y} 0" size="0.01 {tangent} 0.016" '
                f'euler="0 0 {angle}" rgba="0.3 0.5 0.7 1" group="0"/>'
            )
    else:
        for axis in range(2):
            for sign in [-1, 1]:
                pos = [0.0, 0.0, 0.0]
                pos[axis] = sign * (opening + 0.018)
                size = [opening + 0.036, opening, 0.016]
                size[axis], size[1 - axis] = 0.018, opening + 0.036
                geoms.append(
                    f'<geom type="box" pos="{pos[0]} {pos[1]} 0" '
                    f'size="{size[0]} {size[1]} {size[2]}" '
                    'rgba="0.3 0.5 0.7 1" group="0"/>'
                )
    return (
        '<mujoco model="fixture"><worldbody><body><body name="object">' + "".join(geoms) + "</body>"
        '<site name="bottom_site" pos="0 0 -0.016" size="0.001"/>'
        '<site name="top_site" pos="0 0 0.016" size="0.001"/>'
        '<site name="horizontal_radius_site" pos="0.07 0 0" size="0.001"/>'
        "</body></worldbody></mujoco>"
    )


def create_insertion(case, root, controller):
    import numpy as np
    from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
    from robosuite.models.arenas import TableArena
    from robosuite.models.objects import BoxObject, CylinderObject, MujocoXMLObject
    from robosuite.models.tasks import ManipulationTask

    shape = case["task_id"].split("_")[1]
    opening = case.get("opening_m", 0.014)
    xml = root / "fixture.xml"
    xml.write_text(fixture_xml(shape, opening), encoding="utf-8")
    rng = np.random.RandomState(case["layout_seed"])
    peg_xy = np.array([-0.10, -0.12]) + rng.uniform(-0.035, 0.035, 2)
    hole_xy = np.array([0.10, 0.10]) + rng.uniform(-0.035, 0.035, 2)
    yaw = rng.uniform(-0.20, 0.20)
    world = case["physics"]

    class ContactInsertion(SingleArmEnv):
        def _load_model(self):
            super()._load_model()
            self.robots[0].robot_model.set_base_xpos(
                self.robots[0].robot_model.base_xpos_offset["table"](0.8)
            )
            arena = TableArena(table_full_size=(0.8, 0.8, 0.05), table_offset=(0, 0, 0.82))
            arena.set_origin([0, 0, 0])
            args = dict(
                name="peg",
                density=800 * world["mass_scale"],
                friction=[world["friction"], 0.005, 0.0001],
                rgba=[0.85, 0.45, 0.12, 1],
            )
            self.peg = (
                CylinderObject(size=[0.010, 0.05], **args)
                if shape == "round"
                else BoxObject(size=[0.010, 0.010, 0.05], **args)
            )
            self.hole = MujocoXMLObject(str(xml), name="hole", joints=None)
            self.hole.get_obj().set("pos", f"{hole_xy[0]} {hole_xy[1]} 0.836")
            self.model = ManipulationTask(
                arena, [r.robot_model for r in self.robots], [self.peg, self.hole]
            )

        def _setup_references(self):
            super()._setup_references()
            self.obj_body_id = {
                name: self.sim.model.body_name2id(obj.root_body)
                for name, obj in {"peg": self.peg, "hole": self.hole}.items()
            }
            self.objects_dict = {"peg": self.peg, "hole": self.hole}

        def _reset_internal(self):
            super()._reset_internal()
            self.sim.data.set_joint_qpos(
                self.peg.joints[0], np.r_[peg_xy, 0.871, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
            )
            self.sim.forward()

        def _check_success(self):
            peg = self.obj_body_id["peg"]
            axis = self.sim.data.body_xmat[peg].reshape(3, 3)[:, 2]
            hole_top = np.r_[hole_xy, 0.852]
            released = not self._check_grasp(self.robots[0].gripper, self.peg)
            return insertion_success(
                self.sim.data.body_xpos[peg],
                axis,
                hole_top,
                0.05,
                0.010,
                opening,
                released,
                shape,
                self.sim.data.body_xmat[peg].reshape(3, 3),
            )

        def reward(self, action=None):
            return float(self._check_success())

    return ContactInsertion(
        robots="Panda",
        controller_configs=controller,
        initialization_noise=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview"],
        camera_heights=384,
        camera_widths=512,
        control_freq=20,
        horizon=1200,
        ignore_done=True,
        hard_reset=False,
    )


class ManufacturingAdapter:
    def __init__(self, env, case):
        self.env, self.case, self.sim = env, case, env.sim
        self.bind_scene()

    def bind_scene(self):
        env = self.env
        task = self.case["task_id"]
        if task.startswith("nut_"):
            nut = env.nuts[env.nut_id]
            env.objects_dict = {"nut": nut, "shaft": None}
            env.obj_body_id.update(
                {
                    "nut": env.obj_body_id[nut.name],
                    "shaft": env.peg1_body_id if env.nut_id == 0 else env.peg2_body_id,
                }
            )
        elif task == "stack":
            env.objects_dict = {"block": env.cubeA, "base": env.cubeB}
            env.obj_body_id = {"block": env.cubeA_body_id, "base": env.cubeB_body_id}

    def initialize(self, case):
        import numpy as np

        np.random.seed(case["layout_seed"])
        observation = self.env.reset()
        self.bind_scene()
        for _ in range(10):
            observation, _, _, _ = self.env.step([0, 0, 0, 0, 0, 0, -1])
        return observation

    def get_sim_state(self):
        return self.sim.get_state().flatten()

    def check_success(self):
        success = self.env._check_success()
        if self.case["task_id"].startswith("nut_"):
            return bool(
                success
                and not self.env._check_grasp(
                    self.env.robots[0].gripper, self.env.nuts[self.env.nut_id]
                )
            )
        return bool(success)

    def step(self, action):
        import numpy as np

        effective = np.array(action, dtype=float)
        effective[:3] = np.clip(effective[:3] * self.case["physics"]["actuation_gain"], -1, 1)
        return self.env.step(effective)

    def public_observation(self):
        info = {
            "task_family": TASKS[self.case["task_id"]][0],
            "episode_role": "source" if self.case["split"] == "development" else "target",
            "table_z": 0.82,
            "wrist_force_N": self.env.robots[0].ee_force.tolist(),
            "wrist_torque_Nm": self.env.robots[0].ee_torque.tolist(),
            "wrist_sensor_frame": "eef_sensor",
            "object_quat_wxyz": {
                name: self.sim.data.body_xquat[i].tolist()
                for name, i in self.env.obj_body_id.items()
                if name in self.env.objects_dict
            },
        }
        if self.case["task_id"].startswith("nut_"):
            nut = self.env.nuts[self.env.nut_id]
            site = self.sim.model.site_name2id(nut.important_sites["handle"])
            info["handle_pos"] = self.sim.data.site_xpos[site].tolist()
        return info

    def evaluation_metrics(self):
        if self.case["task_id"].startswith("insert_"):
            peg_id = self.env.obj_body_id["peg"]
            pos = self.sim.data.body_xpos[peg_id]
            axis = self.sim.data.body_xmat[peg_id].reshape(3, 3)[:, 2]
            hole = self.sim.data.body_xpos[self.env.obj_body_id["hole"]]
            tip = pos - 0.05 * axis
            return {
                "insertion_depth_m": float(hole[2] + 0.016 - tip[2]),
                "lateral_error_m": math.dist(tip[:2], hole[:2]),
                "axis_up_z": float(axis[2]),
                "tilt_deg": math.degrees(math.acos(float(max(-1, min(1, axis[2]))))),
                "cross_section_margin_m": insertion_fit_margin(
                    pos,
                    axis,
                    [hole[0], hole[1], hole[2] + 0.016],
                    0.010,
                    self.case.get("opening_m", 0.014),
                    self.case["task_id"].split("_")[1],
                    self.sim.data.body_xmat[peg_id].reshape(3, 3),
                ),
                "released": not self.env._check_grasp(self.env.robots[0].gripper, self.env.peg),
            }
        source, destination = (
            ("nut", "shaft") if self.case["task_id"].startswith("nut_") else ("block", "base")
        )
        pos = self.sim.data.body_xpos[self.env.obj_body_id[source]]
        other = self.sim.data.body_xpos[self.env.obj_body_id[destination]]
        obj = self.env.nuts[self.env.nut_id] if source == "nut" else self.env.cubeA
        return {
            "lateral_error_m": math.dist(pos[:2], other[:2]),
            "object_height_m": float(pos[2]),
            "released": not self.env._check_grasp(self.env.robots[0].gripper, obj),
        }

    def close(self):
        self.env.close()


def create_env(case: dict, root: Path):
    if sys.platform == "win32":
        os.environ["MUJOCO_GL"] = "glfw"
    import mujoco
    import numpy as np
    import robosuite
    from robosuite.controllers import load_controller_config

    np.random.seed(case["layout_seed"])
    controller = load_controller_config(default_controller="OSC_POSE")
    task = case["task_id"]
    if task.startswith("insert_"):
        env = create_insertion(case, root, controller)
    else:
        name = (
            "NutAssemblyRound"
            if task == "nut_round"
            else ("NutAssemblySquare" if task == "nut_square" else "Stack")
        )
        env = robosuite.make(
            name,
            robots="Panda",
            controller_configs=controller,
            initialization_noise=None,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            use_object_obs=True,
            camera_names=["agentview"],
            camera_heights=384,
            camera_widths=512,
            horizon=1200,
            ignore_done=True,
            control_freq=20,
            hard_reset=False,
        )
        if task.startswith("nut_"):
            for sampler in env.placement_initializer.samplers.values():
                sampler.rotation = [-0.15, 0.15]
                sampler.x_range = [-0.14, -0.08]
        # A real contact intervention, shared across task families in the same world.
        for obj in env.nuts if task.startswith("nut_") else [env.cubeA, env.cubeB]:
            body = env.sim.model.body_name2id(obj.root_body)
            env.sim.model.body_mass[body] *= case["physics"]["mass_scale"]
            env.sim.model.body_inertia[body] *= case["physics"]["mass_scale"]
            for geom in obj.contact_geoms:
                env.sim.model.geom_friction[env.sim.model.geom_name2id(geom), 0] = case["physics"][
                    "friction"
                ]
    # Equal contact coefficients prevent the other geom's friction from masking the intervention.
    env.sim.model.geom_friction[:, 0] = case["physics"]["friction"]
    if env.sim.model.npair:
        env.sim.model.pair_friction[:, :2] = case["physics"]["friction"]
    mujoco.mj_setConst(env.sim.model._model, env.sim.data._data)
    env.sim.forward()
    physical_state = {
        "friction_intervention": "scene-wide sliding contact coefficient",
        "body_mass": env.sim.model.body_mass.tolist(),
        "body_inertia": env.sim.model.body_inertia.tolist(),
        "geom_friction": env.sim.model.geom_friction.tolist(),
        "pair_friction": env.sim.model.pair_friction.tolist(),
        "timestep": float(env.sim.model.opt.timestep),
    }
    from .spec import write_json

    write_json(root / "physical-state.json", physical_state)
    definition = env.model.get_xml()
    (root / "scene.xml").write_text(definition, encoding="utf-8")
    digest = hashlib.sha256(
        definition.encode()
        + json.dumps(case["physics"], sort_keys=True).encode()
        + json.dumps(physical_state, sort_keys=True).encode()
        + Path(__file__).read_bytes()
    ).hexdigest()
    return ManufacturingAdapter(env, case), SimpleNamespace(language=TASKS[task][1]), None, digest
